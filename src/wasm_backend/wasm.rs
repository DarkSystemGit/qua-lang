use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
};

use binary::{Expr, IntoBytes, WasmVec};

use crate::ast;

use super::{MemPtr, CHECK_TYPES, MEM_PTR_TY};

pub mod binary;

#[derive(Default)]
pub struct Module {
    pub ty_sec: TypeSection,
    // No imports section here b/c it's better to put imports inside the
    // appropriate section (eg funcs section) to make indexing easier.
    pub funcs: Functions,
    pub table_sec: Option<TableSection>,
    pub mem_sec: MemorySection,
    pub globals_sec: GlobalSection,
    pub export_sec: Option<ExportSection>,
    pub start_sec: Option<StartSection>,
    pub elem_sec: Option<ElemSection>,
    pub name_sec: NameSection,
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

                     local_dbg_names: _,
                     next_local_idx: _,
                     stack: _,
                 }| (ty, FuncCode::from((locals, body))),
            )
            .unzip();

        let import_section = ImportSection {
            func_imports: self.funcs.imports.into_iter().collect(),
        };

        let mut buf = Vec::new();
        buf.extend(binary::MAGIC_NUM);
        buf.extend(binary::VERSION);
        buf.extend(self.ty_sec.into_bytes());
        buf.extend(import_section.into_bytes());
        buf.extend(func_sec.into_bytes());
        buf.extend(self.table_sec.into_bytes());
        buf.extend(self.mem_sec.into_bytes());
        buf.extend(self.globals_sec.into_bytes());
        buf.extend(self.export_sec.into_bytes());
        buf.extend(self.start_sec.into_bytes());
        buf.extend(self.elem_sec.into_bytes());
        buf.extend(code_sec.into_bytes());
        buf.extend(self.name_sec.into_bytes());
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

impl FuncType {
    pub fn new(num_args: usize, result_ty: ValType) -> Self {
        let mut params = WasmVec::new();
        params.extend([MEM_PTR_TY]); // self ref
        params.extend(vec![MEM_PTR_TY; num_args]);

        // Functions can only have 1 return type as of now
        let results = [result_ty].into_iter().collect();

        FuncType { params, results }
    }
}

impl IntoBytes for FuncType {
    fn into_bytes(self) -> Vec<u8> {
        let mut buf = vec![binary::TY_FUNC];

        buf.extend(self.params.into_bytes());
        buf.extend(self.results.into_bytes());

        buf
    }
}

#[expect(unused, reason = "not all value types are used in the compiler")]
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
    Ptr,

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
            // A FuncIdx (u32)
            BoxType::Func => 32 / 8,
            // A MemIdx (u32)
            BoxType::Ptr => 32 / 8,
        }
    }

    pub const fn instr_store(self) -> impl IntoBytes {
        match self {
            BoxType::Nil => binary::MEM_I32_STORE_8,
            BoxType::Num => binary::MEM_F64_STORE,
            BoxType::Bool => binary::MEM_I32_STORE_8,
            BoxType::String => binary::MEM_I32_STORE_8,
            BoxType::Func => binary::MEM_I32_STORE,
            BoxType::Ptr => binary::MEM_I32_STORE,
        }
    }

    pub const fn instr_load(self) -> impl IntoBytes {
        match self {
            BoxType::Nil => binary::MEM_I32_LOAD_8U,
            BoxType::Num => binary::MEM_F64_LOAD,
            BoxType::Bool => binary::MEM_I32_LOAD_8U,
            BoxType::String => binary::MEM_I32_LOAD_8U,
            BoxType::Func => binary::MEM_I32_LOAD,
            BoxType::Ptr => binary::MEM_I32_LOAD,
        }
    }

    fn tag(&self) -> u8 {
        match self {
            BoxType::Nil => 0b000,
            BoxType::Num => 0b001,
            BoxType::Bool => 0b010,
            BoxType::String => 0b011,
            BoxType::Func => 0b100,
            BoxType::Ptr => 0b101,
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
            BoxType::Ptr => MEM_PTR_TY,
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
    imports: Vec<FuncImport>,
    funcs: Vec<Func>,
}

impl Functions {
    fn next_idx(&self) -> FuncIdx {
        FuncIdx((self.imports.len() + self.funcs.len()) as u32)
    }

    pub fn insert(
        &mut self,
        func: Func,
        name_sec: &mut NameSection,
        dbg_name: Option<Name>,
    ) -> FuncIdx {
        let idx = self.next_idx();
        // Add debug info
        if let Some(name) = dbg_name {
            name_sec.func(idx, name);
        }
        for (local_idx, name) in &func.local_dbg_names {
            name_sec.local(idx, *local_idx, name.clone());
        }

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

    pub fn imports(&self) -> &Vec<FuncImport> {
        &self.imports
    }

    pub fn all_idxs(&self) -> Vec<FuncIdx> {
        (0..self.funcs.len())
            .map(|i| i + self.imports.len())
            .map(|i| i as u32)
            .map(FuncIdx)
            .collect()
    }

    pub fn raw_func_sec_idx(&self, idx: FuncIdx) -> u32 {
        idx.0 - self.imports.len() as u32
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct FuncIdx(u32);

impl IntoBytes for FuncIdx {
    fn into_bytes(self) -> Vec<u8> {
        self.0.into_bytes()
    }
}

#[derive(Clone, Debug)]
pub struct FuncImport {
    pub module: Name,
    pub name: Name,
    pub ty: TypeIdx,
}

impl FuncImport {
    pub fn dbg_name(&self) -> Name {
        Name(format!("{}.{}", self.module.0, self.name.0))
    }
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

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
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
    local_dbg_names: HashMap<LocalIdx, Name>,
    next_local_idx: u32,
    stack: HashMap<ast::IdentLocation, LocalIdx>,

    pub body: binary::Expr,
}

impl Func {
    pub fn new(
        ty: TypeIdx,
        dbg_name: Option<Name>,
        arguments: impl IntoIterator<Item = Option<ast::Identifier>>,
        upvalues: &[ast::Upvalue],
    ) -> Self {
        // Transform into Iterator<Item = Option<Name>>
        let arguments = arguments
            .into_iter()
            .map(|ident| ident.map(|ident| Name(ident.name)));
        // Add in self ref
        let arguments = [dbg_name].into_iter().chain(arguments);

        let mut this = Func::new_base(ty, arguments);

        // Load upvalues
        // The base ptr is stored in the boxed self ptr
        let self_ptr = MemPtr::func_self_ref(upvalues.len() as u32);
        // Start at 1 b/c the func index is first
        for (upvalue, offset) in upvalues.iter().zip(1..=upvalues.len() as u32) {
            self_ptr.gen_offset(&mut this, offset);

            // `ptr` has type `BoxType::Ptr` (ie it is a pointer to a pointer)
            this.gen_unbox_no_tag(BoxType::Ptr);

            let i = offset - 1;
            let loc = ast::IdentLocation::Upvalue(ast::UpvalueIndex(i as usize));
            this.gen_local_set(MEM_PTR_TY, Some(loc), Some(Name(upvalue.dbg_name.clone())));
        }

        this
    }

    pub fn new_base(ty: TypeIdx, arguments: impl IntoIterator<Item = Option<Name>>) -> Self {
        let mut this = Func {
            ty,
            locals: Vec::new(),
            local_dbg_names: HashMap::new(),
            next_local_idx: 0,
            stack: HashMap::new(),
            body: binary::Expr::new(),
        };

        // First the args
        // - Don't actual make a locals entry for them, because they are
        //   implicitly declared.
        for name in arguments {
            let loc = ast::IdentLocation::Stack(ast::StackIndex(this.next_local_idx as usize));
            let idx = LocalIdx(this.next_local_idx);
            this.next_local_idx += 1;

            this.stack.insert(loc, idx);

            if let Some(name) = name {
                this.local_dbg_names.insert(idx, name);
            }
        }

        this
    }

    pub fn insert_local(
        &mut self,
        ty: ValType,
        stack_loc: Option<ast::IdentLocation>,
        dbg_name: Option<Name>,
    ) -> LocalIdx {
        match self.locals.last_mut() {
            Some(local) if local.ty == ty => local.num += 1,
            Some(_) | None => self.locals.push(Local::new(ty)),
        }

        let idx = LocalIdx(self.next_local_idx);
        self.next_local_idx += 1;

        if let Some(stack_loc) = stack_loc {
            self.stack.insert(stack_loc, idx);
        }

        if let Some(name) = dbg_name {
            self.local_dbg_names.insert(idx, name);
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
        dbg_name: Option<Name>,
    ) -> LocalIdx {
        let idx = self.insert_local(ty, stack_loc, dbg_name);
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
        dbg_name: Option<Name>,
    ) -> LocalIdx {
        let idx = self.insert_local(ty, stack_loc, dbg_name);
        self.body.extend(binary::LOCAL_TEE);
        self.body.extend(idx);
        idx
    }

    /// Reads from a local and places it on the stack.
    ///
    /// `[] -> [T]`
    pub fn gen_local_get(&mut self, idx: LocalIdx) {
        assert!(
            idx.0 < self.next_local_idx,
            "tried to get local {:#?} but next_local_idx is {:#?}",
            idx.0,
            self.next_local_idx
        );
        self.body.extend(binary::LOCAL_GET);
        self.body.extend(idx);
    }

    pub fn gen_stack_get(&mut self, stack_loc: &ast::IdentLocation) {
        let idx = *self
            .stack
            .get(stack_loc)
            .unwrap_or_else(|| panic!("stack location should be valid {stack_loc:?}"));
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
        if ptr.includes_tag_byte {
            // Write the memory location
            ptr.gen_load(self);

            // Write tag byte
            self.body.extend(binary::CONST_I32);
            self.body.extend(ptr.box_ty.tag());

            // Store in memory
            self.body.extend(binary::MEM_I32_STORE_8);
            self.body.extend([
                0x00u8, // Align 2^0=1
                0x00,   // Offset 0
            ]);
        }

        for (i, gen_value) in gen_values.into_iter().enumerate() {
            // Write the memory location
            ptr.gen_offset(self, i as u32);

            // Generate the value to store
            gen_value(self);

            // Store that value in memory
            self.body.extend(ptr.box_ty.instr_store());
            self.body.extend([
                0x00u8, // Align 2^0=1
                0x00,   // Offset 0
            ]);
        }

        // Return a pointer to the memory location
        ptr.gen_load(self)
    }

    /// Unboxes the pointer on top of the stack and returns the value.
    ///
    /// `[I32] -> [T]`
    pub fn gen_unbox(&mut self, box_ty: BoxType) {
        if CHECK_TYPES {
            // Save address to local
            let ptr_idx = self.gen_local_tee(MEM_PTR_TY, None, None);

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

            // Put address back on the stack
            self.gen_local_get(ptr_idx);
        }

        // Add one to ignore the tag byte
        self.body.extend([binary::CONST_I32, 0x01, binary::ADD_I32]);
        self.gen_unbox_no_tag(box_ty);
    }

    /// Reads the pointer on top of the stack, when the pointer is pointing to
    /// the actual data and not the tag byte.
    pub fn gen_unbox_no_tag(&mut self, box_ty: BoxType) {
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
        let res_idx = self.gen_local_set(rebox_ptr.box_ty.into(), None, None);

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

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct LocalIdx(u32);
impl LocalIdx {
    pub const FUNC_SELF_REF: Self = LocalIdx(0);
}
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

#[expect(unused, reason = "not all ref types are used in the compiler")]
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

#[derive(Debug, Default)]
pub struct GlobalSection {
    pub globals: WasmVec<Global>,
}

impl GlobalSection {
    pub fn insert(
        &mut self,
        global: Global,
        dbg_info: Option<(&mut NameSection, Name)>,
    ) -> GlobalIdx {
        let idx = GlobalIdx(self.globals.size());

        self.globals.extend([global]);

        if let Some((name_sec, name)) = dbg_info {
            name_sec.global(idx, name);
        }

        idx
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct GlobalIdx(u32);

impl IntoBytes for GlobalIdx {
    fn into_bytes(self) -> Vec<u8> {
        self.0.into_bytes()
    }
}

impl IntoBytes for GlobalSection {
    fn into_bytes(self) -> Vec<u8> {
        binary::sec_bytes(binary::SEC_GLOBAL, self.globals)
    }
}

#[derive(Debug)]
pub struct Global {
    pub ty: ValType,
    pub mutable: bool,
    pub init: Expr,
}

impl IntoBytes for Global {
    fn into_bytes(self) -> Vec<u8> {
        let mut buf = self.ty.into_bytes();
        buf.extend(self.mutable.into_bytes());
        buf.extend(self.init.into_bytes());
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

#[derive(Default)]
pub struct NameSection {
    module: Option<Name>,
    funcs: NameMap<FuncIdx>,
    locals: NameMap<FuncIdx, NameMap<LocalIdx>>,
    globals: NameMap<GlobalIdx>,
}

impl NameSection {
    #[expect(unused, reason = "module doesn't need to be named")]
    pub fn module(&mut self, name: Name) {
        self.module = Some(name);
    }

    pub fn func(&mut self, idx: FuncIdx, name: Name) {
        self.funcs.insert(idx, name);
    }

    pub fn local(&mut self, func_idx: FuncIdx, local_idx: LocalIdx, name: Name) {
        let map = self.locals.entry(func_idx).or_default();
        map.insert(local_idx, name);
    }

    pub fn global(&mut self, idx: GlobalIdx, name: Name) {
        self.globals.insert(idx, name);
    }
}

impl IntoBytes for NameSection {
    fn into_bytes(self) -> Vec<u8> {
        let mut buf = Name("name".to_string()).into_bytes();

        const SUBSEC_MODULE: u8 = 0x00;
        const SUBSEC_FUNCS: u8 = 0x01;
        const SUBSEC_LOCALS: u8 = 0x02;
        const SUBSEC_GLOBALS: u8 = 0x07; // See <https://github.com/WebAssembly/extended-name-section/blob/main/proposals/extended-name-section/Overview.md#global-names>

        if let Some(module) = self.module {
            buf.extend(binary::sec_bytes(SUBSEC_MODULE, module));
        }
        buf.extend(binary::sec_bytes(SUBSEC_FUNCS, self.funcs));
        buf.extend(binary::sec_bytes(SUBSEC_LOCALS, self.locals));
        buf.extend(binary::sec_bytes(SUBSEC_GLOBALS, self.globals));

        binary::sec_bytes(binary::SEC_CUSTOM, buf)
    }
}

#[derive(Debug)]
pub struct NameMap<K, V = Name> {
    assoc: HashMap<K, V>,
}

impl<K: IntoBytes + Ord, V: IntoBytes> IntoBytes for NameMap<K, V> {
    fn into_bytes(self) -> Vec<u8> {
        let mut assoc = self.assoc.into_iter().collect::<Vec<(K, V)>>();
        // indices must be in order
        assoc.sort_by(|(a, _), (b, _)| a.cmp(b));

        // binary rep is a wasm vec
        assoc.into_iter().collect::<WasmVec<(K, V)>>().into_bytes()
    }
}

impl<K, V> Deref for NameMap<K, V> {
    type Target = HashMap<K, V>;

    fn deref(&self) -> &Self::Target {
        &self.assoc
    }
}
impl<K, V> DerefMut for NameMap<K, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.assoc
    }
}

impl<K, V> Default for NameMap<K, V> {
    fn default() -> Self {
        NameMap {
            assoc: HashMap::default(),
        }
    }
}
