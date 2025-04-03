use std::collections::HashMap;

use binary::{Expr, IntoBytes, WasmVec};

use crate::ast;

use super::{MemPtr, MEM_PTR_TY};

pub mod binary;

#[derive(Default)]
pub struct Module {
    pub ty_sec: TypeSection,
    // No imports section here b/c it's better to put imports inside the
    // appropriate section (eg funcs section) to make indexing easier.
    pub funcs: Functions,
    pub table_sec: Option<TableSection>,
    pub mem_sec: MemorySection,
    pub export_sec: Option<ExportSection>,
    pub start_sec: Option<StartSection>,
    pub elem_sec: Option<ElemSection>,
}

impl IntoBytes for Module {
    fn into_bytes(self) -> Vec<u8> {
        let (func_sec, code_sec): (FunctionSection, CodeSection) = self
            .funcs
            .funcs
            .into_iter()
            .map(
                |Func {
                     ty,
                     locals,
                     body,

                     next_local_idx: _,
                     stack: _,
                 }| (ty, FuncCode::from((locals, body))),
            )
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
        buf.extend(self.table_sec.into_bytes());
        buf.extend(self.mem_sec.into_bytes());
        buf.extend(self.export_sec.into_bytes());
        buf.extend(self.start_sec.into_bytes());
        buf.extend(self.elem_sec.into_bytes());
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

/// The type of value to store in memory.
///
/// # Binary Format
/// - The 3 least significant bits of the first byte (the one that is pointed
///   to) contains a tag indicating the type of the data.
#[derive(Clone, Copy, Debug)]
pub enum BoxType {
    Nil,
    Num,
    Bool,
    String,
    Func,
}

impl BoxType {
    pub const fn size(self) -> u32 {
        match self {
            BoxType::Nil => 1,
            BoxType::Num => 64 / 8,
            BoxType::Bool => 1,
            // A string is stored as bytes
            BoxType::String => 1,
            // Just a FuncIdx. an array of bytes.
            BoxType::Func => 1,
        }
    }

    pub const fn instr_store(self) -> impl IntoBytes {
        match self {
            BoxType::Nil => binary::MEM_I32_STORE_8,
            BoxType::Num => binary::MEM_F64_STORE,
            BoxType::Bool => binary::MEM_I32_STORE_8,
            BoxType::String => binary::MEM_I32_STORE_8,
            BoxType::Func => binary::MEM_I32_STORE_8,
        }
    }

    pub const fn instr_load(self) -> impl IntoBytes {
        match self {
            BoxType::Nil => binary::MEM_I32_LOAD_8U,
            BoxType::Num => binary::MEM_F64_LOAD,
            BoxType::Bool => binary::MEM_I32_LOAD_8U,
            BoxType::String => binary::MEM_I32_LOAD_8U,
            BoxType::Func => binary::MEM_I32_LOAD_8U,
        }
    }

    fn tag(&self) -> u8 {
        match self {
            BoxType::Nil => 0b000,
            BoxType::Num => 0b001,
            BoxType::Bool => 0b010,
            BoxType::String => 0b011,
            BoxType::Func => 0b100,
        }
    }
}

impl From<BoxType> for ValType {
    fn from(value: BoxType) -> Self {
        match value {
            BoxType::Nil => ValType::I32,
            BoxType::Num => ValType::F64,
            BoxType::Bool => ValType::I32,
            BoxType::String => ValType::I32,
            BoxType::Func => ValType::I32,
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
    locals: WasmVec<Local>,
    expr: binary::Expr,
}

impl From<(Vec<Local>, binary::Expr)> for FuncCode {
    fn from(value: (Vec<Local>, binary::Expr)) -> Self {
        FuncCode {
            locals: value.0.into_iter().collect(),
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

    pub fn all_idxs(&self) -> Vec<FuncIdx> {
        (0..self.funcs.len())
            .map(|i| i as u32 + self.imports.size())
            .map(FuncIdx)
            .collect()
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

    locals: Vec<Local>,
    next_local_idx: u32,
    stack: HashMap<ast::IdentLocation, LocalIdx>,

    pub body: binary::Expr,
}

impl Func {
    pub fn new(ty: TypeIdx) -> Self {
        Func {
            ty,
            locals: Vec::new(),
            next_local_idx: 0,
            stack: HashMap::new(),
            body: binary::Expr::new(),
        }
    }

    pub fn insert_local(&mut self, ty: ValType, stack_loc: Option<ast::IdentLocation>) -> LocalIdx {
        match self.locals.last_mut() {
            Some(local) if local.ty == ty => local.num += 1,
            Some(_) | None => self.locals.push(Local::new(ty)),
        }

        let idx = LocalIdx(self.next_local_idx);
        self.next_local_idx += 1;

        if let Some(stack_loc) = stack_loc {
            self.stack.insert(stack_loc, idx);
        }

        idx
    }

    /// Places the value on the top of the stack into a new local.
    ///
    /// `[T] -> []`
    pub fn gen_local_set(
        &mut self,
        ty: ValType,
        stack_loc: Option<ast::IdentLocation>,
    ) -> LocalIdx {
        let idx = self.insert_local(ty, stack_loc);
        self.body.extend(binary::LOCAL_SET);
        self.body.extend(idx);
        idx
    }

    /// Places the value on the top of the stack in a new local and returns it.
    ///
    /// `[T] -> [T]`
    pub fn gen_local_tee(
        &mut self,
        ty: ValType,
        stack_loc: Option<ast::IdentLocation>,
    ) -> LocalIdx {
        let idx = self.insert_local(ty, stack_loc);
        self.body.extend(binary::LOCAL_TEE);
        self.body.extend(idx);
        idx
    }

    /// Reads from a local and places it on the stack.
    ///
    /// `[] -> [T]`
    pub fn gen_local_get(&mut self, idx: LocalIdx) {
        self.body.extend(binary::LOCAL_GET);
        self.body.extend(idx);
    }

    pub fn gen_stack_get(&mut self, stack_loc: &ast::IdentLocation) {
        let idx = *self.stack.get(stack_loc).expect("stack location is valid");
        self.gen_local_get(idx);
    }

    /// Boxes the top item on the stack and returns a pointer to it.
    ///
    /// `[T] -> [I32]`
    pub fn gen_box<T: FnOnce(&mut Self), I: IntoIterator<Item = T>>(
        &mut self,
        ptr: MemPtr,
        gen_values: I,
        // gen_values: &[impl Fn(&mut WasmGenState)],
    ) where
        I::IntoIter: ExactSizeIterator,
    {
        let mut i: u32 = 0;
        if ptr.includes_tag_byte {
            // Write the memory location
            self.body.extend(binary::CONST_I32);
            self.body.extend(ptr.offset(i));

            // Write tag byte
            self.body.extend(binary::CONST_I32);
            self.body.extend(ptr.box_ty.tag());

            // Store in memory
            self.body.extend(binary::MEM_I32_STORE_8);
            self.body.extend([
                0x00u8, // Align 2^0=1
                0x00,   // Offset 0
            ]);

            i += 1;
        }

        for gen_value in gen_values.into_iter() {
            // Write the memory location
            self.body.extend(binary::CONST_I32);
            self.body.extend(ptr.offset(i));

            // Generate the value to store
            gen_value(self);

            // Store that value in memory
            self.body.extend(ptr.box_ty.instr_store());
            self.body.extend([
                0x00u8, // Align 2^0=1
                0x00,   // Offset 0
            ]);

            i += 1;
        }

        // Return a pointer to the memory location
        self.body.extend(binary::CONST_I32);
        self.body.extend([ptr]);
    }

    /// Unboxes the pointer on top of the stack and returns the value.
    ///
    /// `[I32] -> [T]`
    pub fn gen_unbox(&mut self, box_ty: BoxType) {
        // Save address to local
        let ptr_idx = self.gen_local_tee(MEM_PTR_TY, None);

        self.body.extend(binary::MEM_I32_LOAD_8U);
        self.body.extend([
            0x00u8, // Align 2^0=1
            0x00,   // Offset 0
        ]);

        // Verify type
        // Tag is on top of stack
        self.body.extend([binary::CONST_I32, box_ty.tag()]);
        self.body.extend(binary::NE_I32);
        self.body
            .extend([binary::IF, binary::TY_NEVER, binary::TRAP, binary::END]);

        // Get the address
        self.gen_local_get(ptr_idx);
        // Add one to ignore the tag byte
        self.body.extend([binary::CONST_I32, 0x01, binary::ADD_I32]);
        // Actually load the data
        self.body.extend(box_ty.instr_load());
        self.body.extend([
            0x00u8, // Align 2^0=1
            0x00,   // Offset 0
        ]);
    }

    /// Unboxes the pointer on top of the stack, runs `func()` to work with it,
    /// and then reboxes the top of the stack.
    ///
    /// `[I32] -> [I32]`
    ///
    /// # Parameters
    /// - `mem_unbox`: What to unbox into before `func()` is run.
    /// - `mem_rebox`: What to rebox into after `func()` is run.
    /// - `func`: The function to run w/ the unboxed value. Must be `[Unbox] -> [Rebox]`.
    pub fn unwrap_box(
        &mut self,
        unbox_ty: BoxType,
        rebox_ptr: MemPtr,
        func: impl FnOnce(&mut Self),
    ) {
        self.gen_unbox(unbox_ty);

        func(self);
        let res_idx = self.gen_local_set(rebox_ptr.box_ty.into(), None);

        self.gen_box(
            rebox_ptr,
            [|func: &mut Self| {
                func.gen_local_get(res_idx);
            }],
        );
    }
}

#[derive(Debug)]
struct Local {
    num: u32,
    ty: ValType,
}

impl Local {
    fn new(ty: ValType) -> Self {
        Local { num: 1, ty }
    }
}

impl IntoBytes for Local {
    fn into_bytes(self) -> Vec<u8> {
        let mut buf = self.num.into_bytes();
        buf.extend(self.ty.into_bytes());
        buf
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LocalIdx(u32);
impl IntoBytes for LocalIdx {
    fn into_bytes(self) -> Vec<u8> {
        self.0.into_bytes()
    }
}

#[derive(Default)]
pub struct TableSection {
    tables: WasmVec<TableType>,
}

impl TableSection {
    pub fn new() -> Self {
        TableSection {
            tables: WasmVec::new(),
        }
    }

    pub fn insert(&mut self, table: TableType) -> TableIdx {
        let idx = TableIdx(self.tables.size());
        self.tables.extend([table]);
        idx
    }
}

impl IntoBytes for TableSection {
    fn into_bytes(self) -> Vec<u8> {
        binary::sec_bytes(binary::SEC_TABLE, self.tables)
    }
}

#[derive(Debug)]
pub struct TableIdx(u32);
impl IntoBytes for TableIdx {
    fn into_bytes(self) -> Vec<u8> {
        self.0.into_bytes()
    }
}

#[derive(Debug)]
pub struct TableType {
    pub ty: RefType,
    pub limits: Limits,
}

impl IntoBytes for TableType {
    fn into_bytes(self) -> Vec<u8> {
        let mut buf = self.ty.into_bytes();
        buf.extend(self.limits.into_bytes());
        buf
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum RefType {
    Func,
    Extern,
}

impl IntoBytes for RefType {
    fn into_bytes(self) -> Vec<u8> {
        match self {
            RefType::Func => vec![binary::TY_FUNC_REF],
            RefType::Extern => vec![binary::TY_EXTERN_REF],
        }
    }
}

#[derive(Default)]
pub struct ElemSection {
    segments: WasmVec<Elem>,
}

impl ElemSection {
    pub fn new() -> Self {
        ElemSection {
            segments: WasmVec::new(),
        }
    }
    pub fn insert(&mut self, segment: Elem) {
        self.segments.extend([segment]);
    }
}

impl IntoBytes for ElemSection {
    fn into_bytes(self) -> Vec<u8> {
        binary::sec_bytes(binary::SEC_ELEM, self.segments)
    }
}

#[derive(Debug)]
pub struct Elem {
    pub ty: RefType,
    pub init: WasmVec<FuncIdx>,
    pub mode: ElemMode,
}

impl Elem {
    pub fn insert(&mut self, func: Vec<FuncIdx>) {
        self.init.extend(func);
    }
}

impl Default for Elem {
    fn default() -> Self {
        Elem {
            ty: RefType::Func,
            init: WasmVec::default(),
            mode: ElemMode::default(),
        }
    }
}

impl IntoBytes for Elem {
    /// Described in <https://webassembly.github.io/spec/core/binary/modules.html#element-section>.
    /// I'm not sure I fully understand it, so I'm just implementing the subset
    /// that is useful to this program.
    fn into_bytes(self) -> Vec<u8> {
        let ElemMode::Active {
            table: table_idx,
            offset,
        } = self.mode;

        assert_eq!(self.ty, RefType::Func);
        assert_eq!(table_idx.0, 0);

        let mut buf = vec![0x00];

        buf.extend(offset.into_bytes());
        buf.extend(self.init.into_bytes());

        buf
    }
}

#[derive(Debug)]
pub enum ElemMode {
    // Not all the possibilities, but the other ones start to get more complex
    // to encode.
    Active { table: TableIdx, offset: Expr },
}

impl Default for ElemMode {
    fn default() -> Self {
        ElemMode::Active {
            table: TableIdx(0),
            offset: {
                let mut expr = Expr::new();
                expr.extend([binary::CONST_I32, 0x00]);
                expr
            },
        }
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
