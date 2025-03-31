use std::collections::HashMap;

use binary::{IntoBytes, WasmVec};

pub mod binary;

#[derive(Default)]
pub struct Module {
    pub ty_sec: TypeSection,
    // No imports section here b/c it's better to put imports inside the
    // appropriate section (eg funcs section) to make indexing easier.
    pub funcs: Functions,
    pub mem_sec: MemorySection,
    pub export_sec: Option<ExportSection>,
    pub start_sec: Option<StartSection>,
}

impl IntoBytes for Module {
    fn into_bytes(self) -> Vec<u8> {
        let (func_sec, code_sec): (FunctionSection, CodeSection) = self
            .funcs
            .funcs
            .into_iter()
            .map(|Func { ty, locals, body }| (ty, FuncCode::from((locals, body))))
            .unzip();

        let import_section = ImportSection {
            func_imports: self.funcs.imports,
        };

        let mut buf = Vec::new();
        buf.extend(binary::MAGIC_NUM);
        buf.extend(binary::VERSION);
        buf.extend(self.ty_sec.into_bytes());
        buf.extend(import_section.into_bytes());
        buf.extend(func_sec.into_bytes());
        buf.extend(self.mem_sec.into_bytes());
        buf.extend(self.export_sec.into_bytes());
        buf.extend(self.start_sec.into_bytes());
        buf.extend(code_sec.into_bytes());
        buf
    }
}

#[derive(Debug, Default)]
pub struct TypeSection {
    types_map: HashMap<FuncType, TypeIdx>,
    types: WasmVec<FuncType>,
}

impl TypeSection {
    pub fn insert(&mut self, ty: FuncType) -> TypeIdx {
        if let Some(idx) = self.types_map.get(&ty) {
            *idx
        } else {
            let next_idx = self.types.size();
            self.types.extend([ty.clone()]);

            let idx = TypeIdx(next_idx);
            self.types_map.insert(ty, idx);

            idx
        }
    }
}

impl IntoBytes for TypeSection {
    fn into_bytes(self) -> Vec<u8> {
        binary::sec_bytes(binary::SEC_TY, self.types)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TypeIdx(u32);

impl IntoBytes for TypeIdx {
    fn into_bytes(self) -> Vec<u8> {
        self.0.into_bytes()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FuncType {
    pub params: WasmVec<ValType>,
    pub results: WasmVec<ValType>,
}

impl IntoBytes for FuncType {
    fn into_bytes(self) -> Vec<u8> {
        let mut buf = vec![binary::TY_FUNC];

        buf.extend(self.params.into_bytes());
        buf.extend(self.results.into_bytes());

        buf
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValType {
    I32,
    I64,
    F32,
    F64,
}

impl IntoBytes for ValType {
    fn into_bytes(self) -> Vec<u8> {
        match self {
            ValType::I32 => vec![0x7F],
            ValType::I64 => vec![0x7E],
            ValType::F32 => vec![0x7D],
            ValType::F64 => vec![0x7C],
        }
    }
}

#[derive(Debug)]
struct ImportSection {
    pub func_imports: WasmVec<FuncImport>,
}

impl IntoBytes for ImportSection {
    fn into_bytes(self) -> Vec<u8> {
        binary::sec_bytes(binary::SEC_IMPORT, self.func_imports)
    }
}

#[derive(Debug, Default)]
struct FunctionSection {
    funcs: WasmVec<TypeIdx>,
}

impl Extend<TypeIdx> for FunctionSection {
    fn extend<T: IntoIterator<Item = TypeIdx>>(&mut self, iter: T) {
        self.funcs.extend(iter)
    }
}

impl IntoBytes for FunctionSection {
    fn into_bytes(self) -> Vec<u8> {
        binary::sec_bytes(binary::SEC_FUNC, self.funcs)
    }
}

#[derive(Debug, Default)]
struct CodeSection {
    codes: WasmVec<FuncCode>,
}

impl Extend<FuncCode> for CodeSection {
    fn extend<T: IntoIterator<Item = FuncCode>>(&mut self, iter: T) {
        self.codes.extend(iter)
    }
}

impl IntoBytes for CodeSection {
    fn into_bytes(self) -> Vec<u8> {
        binary::sec_bytes(binary::SEC_CODE, self.codes)
    }
}

#[derive(Debug)]
struct FuncCode {
    locals: WasmVec<ValType>,
    expr: binary::Expr,
}

impl From<(WasmVec<ValType>, binary::Expr)> for FuncCode {
    fn from(value: (WasmVec<ValType>, binary::Expr)) -> Self {
        FuncCode {
            locals: value.0,
            expr: value.1,
        }
    }
}

impl IntoBytes for FuncCode {
    fn into_bytes(self) -> Vec<u8> {
        let mut code = self.locals.into_bytes();
        code.extend(self.expr.into_bytes());

        // Write size header
        let size = (code.len() as u32).into_bytes();
        code.splice(..0, size);

        code
    }
}

/// *Not* the functions section. It is a merge of the function, code, and
/// (function) imports section.
#[derive(Default)]
pub struct Functions {
    imports: WasmVec<FuncImport>,
    funcs: Vec<Func>,
}

impl Functions {
    fn next_idx(&self) -> FuncIdx {
        FuncIdx(self.imports.size() + self.funcs.len() as u32)
    }

    pub fn insert(&mut self, func: Func) -> FuncIdx {
        let idx = self.next_idx();
        self.funcs.push(func);
        idx
    }

    pub fn insert_import(&mut self, import: FuncImport) -> FuncIdx {
        if !self.funcs.is_empty() {
            // Don't continue, because it would mess up any stored indexes.
            panic!("Tried to add import when `funcs` was non-empty.");
        }

        let idx = self.next_idx();
        self.imports.extend([import]);
        idx
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FuncIdx(u32);

impl IntoBytes for FuncIdx {
    fn into_bytes(self) -> Vec<u8> {
        self.0.into_bytes()
    }
}

#[derive(Debug)]
pub struct FuncImport {
    pub module: Name,
    pub name: Name,
    pub ty: TypeIdx,
}

impl IntoBytes for FuncImport {
    fn into_bytes(self) -> Vec<u8> {
        let mut buf = Vec::new();

        buf.extend(self.module.into_bytes());
        buf.extend(self.name.into_bytes());

        buf.push(0x00);
        buf.extend(self.ty.into_bytes());

        buf
    }
}

#[derive(Debug)]
pub struct Name(pub String);
impl IntoBytes for Name {
    fn into_bytes(self) -> Vec<u8> {
        let buf = WasmVec::from_iter(self.0.bytes());
        buf.into_bytes()
    }
}

pub struct Func {
    pub ty: TypeIdx,
    locals: WasmVec<ValType>,
    pub body: binary::Expr,
}

impl Func {
    pub fn new(ty: TypeIdx) -> Self {
        Func {
            ty,
            locals: WasmVec::new(),
            body: binary::Expr::new(),
        }
    }

    pub fn insert_local(&mut self, ty: ValType) -> LocalIdx {
        let idx = LocalIdx(self.locals.size());
        self.locals.extend([ty]);
        idx
    }
}

pub struct LocalIdx(u32);
impl IntoBytes for LocalIdx {
    fn into_bytes(self) -> Vec<u8> {
        self.0.into_bytes()
    }
}

#[derive(Default)]
pub struct MemorySection {
    memories: WasmVec<MemType>,
}

impl MemorySection {
    pub fn insert(&mut self, mem: MemType) -> MemIdx {
        let idx = MemIdx(self.memories.size());
        self.memories.extend([mem]);
        idx
    }
}

impl IntoBytes for MemorySection {
    fn into_bytes(self) -> Vec<u8> {
        binary::sec_bytes(binary::SEC_MEM, self.memories)
    }
}

#[derive(Debug)]
pub struct MemIdx(u32);
impl IntoBytes for MemIdx {
    fn into_bytes(self) -> Vec<u8> {
        self.0.into_bytes()
    }
}

pub struct MemType {
    // The min and max size, in multiples of the page size.
    pub limits: Limits,
}

impl IntoBytes for MemType {
    fn into_bytes(self) -> Vec<u8> {
        self.limits.into_bytes()
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Limits {
    pub min: u32,
    pub max: Option<u32>,
}

impl IntoBytes for Limits {
    fn into_bytes(self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Header
        buf.push(if self.max.is_some() { 0x01 } else { 0x00 });
        // Min value
        buf.extend(self.min.into_bytes());
        // Max value if it exists
        if let Some(max) = self.max {
            buf.extend(max.into_bytes());
        }

        buf
    }
}

#[derive(Debug)]
pub struct StartSection {
    pub func: FuncIdx,
}

impl IntoBytes for StartSection {
    fn into_bytes(self) -> Vec<u8> {
        binary::sec_bytes(binary::SEC_START, self.func)
    }
}

#[derive(Debug)]
pub struct ExportSection {
    exports: WasmVec<Export>,
}

impl ExportSection {
    pub fn new() -> Self {
        ExportSection {
            exports: WasmVec::new(),
        }
    }

    pub fn insert(&mut self, export: Export) {
        self.exports.extend([export]);
    }
}

impl IntoBytes for ExportSection {
    fn into_bytes(self) -> Vec<u8> {
        binary::sec_bytes(binary::SEC_EXPORT, self.exports)
    }
}

#[derive(Debug)]
pub struct Export {
    pub name: Name,
    pub desc: ExportDesc,
}

impl IntoBytes for Export {
    fn into_bytes(self) -> Vec<u8> {
        let mut buf = self.name.into_bytes();
        buf.extend(self.desc.into_bytes());
        buf
    }
}

#[derive(Debug)]
pub enum ExportDesc {
    Func(FuncIdx),
    // Table(TableIdx), // Uncomment when tables exist
    Mem(MemIdx),
    // Global(GlobalIdx), // Uncomment when globals exist
}

impl IntoBytes for ExportDesc {
    fn into_bytes(self) -> Vec<u8> {
        let (discriminant, contents) = match self {
            ExportDesc::Func(func_idx) => (0x00, func_idx.into_bytes()),
            ExportDesc::Mem(mem_idx) => (0x02, mem_idx.into_bytes()),
        };
        let mut buf = vec![discriminant];
        buf.extend(contents);
        buf
    }
}
