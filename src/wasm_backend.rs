use wasm::binary::{IntoBytes, WasmVec};

use crate::ast;

pub fn gen_wasm(program: ast::Program) -> Vec<u8> {
    let mut state = WasmGenState::new();
    state.gen_program(program);
    state.finish()
}

struct WasmGenState {
    module: wasm::Module,
    cur_func: wasm::Func,
    mem_store: MemStore,
}
impl WasmGenState {
    fn new() -> Self {
        let mut module = wasm::Module::default();

        let import_host_print = wasm::FuncImport {
            module: wasm::Name("host".to_string()),
            name: wasm::Name("print".to_string()),
            ty: {
                let ty = wasm::FuncType {
                    // TODO: funcs
                    params: [wasm::ValType::F64].into_iter().collect(),
                    results: WasmVec::new(),
                };
                module.ty_sec.insert(ty)
            },
        };
        module.funcs.insert_import(import_host_print);

        // The idx is always 0, at least until wasm supports multiple memories
        let _ = module.mem_sec.insert(wasm::MemType {
            limits: wasm::Limits { min: 64, max: None },
        });

        let main_func = {
            let ty = wasm::FuncType {
                params: WasmVec::new(),
                results: WasmVec::new(),
            };
            let ty = module.ty_sec.insert(ty);
            wasm::Func::new(ty)
        };

        WasmGenState {
            module,
            cur_func: main_func,
            mem_store: MemStore::new(),
        }
    }

    fn finish(mut self) -> Vec<u8> {
        // Set up main function
        let idx = self.module.funcs.insert(self.cur_func);
        let start_sec = wasm::StartSection { func: idx };
        self.module.start_sec = Some(start_sec);

        self.module.into_bytes()
    }

    fn gen_program(&mut self, program: ast::Program) {
        for stmt in program {
            self.gen_stmt(stmt);
        }
    }

    fn gen_stmt(&mut self, stmt: ast::Stmt) {
        match stmt {
            ast::Stmt::Let(binding) => self.gen_binding(binding),
            ast::Stmt::Expr(expr) => self.gen_expr(expr),
        }
    }

    fn gen_binding(&mut self, binding: ast::Binding) {
        match binding.metadata {
            ast::BindingMetadata::Var => {
                self.gen_expr(binding.value);

                let idx = self.cur_func.insert_local(wasm::ValType::I32);
                self.cur_func.body.extend(wasm::binary::LOCAL_SET);
                self.cur_func.body.extend(idx);
            }
            ast::BindingMetadata::Func {
                arguments,
                upvalues,
            } => {
                // Everything is boxed (ie an I32)
                let param_tys = arguments.iter().map(|_| wasm::ValType::I32).collect();
                let result_ty = wasm::ValType::I32;
                let ty = wasm::FuncType {
                    params: param_tys,
                    // Functions can only have 1 return type as of now
                    results: [result_ty].into_iter().collect(),
                };

                let ty = self.module.ty_sec.insert(ty);
                let func = wasm::Func::new(ty);

                // TODO: actual generate body

                self.module.funcs.insert(func);
            }
        }
    }

    fn gen_expr(&mut self, expr: ast::Expr) {
        match expr {
            ast::Expr::Block(block) => {
                for stmt in block.stmts {
                    self.gen_stmt(stmt);
                }

                if let Some(return_expr) = block.return_expr {
                    self.gen_expr(*return_expr);
                }
            }
            ast::Expr::Call(call) => {
                // Put arguments onto the stack
                for arg in call.arguments {
                    self.gen_expr(arg);
                    // Unbox for now so printing works nicely
                    // TODO: funcs
                    self.gen_unbox(MemBox::F64);
                }

                // TODO: funcs
                self.cur_func.body.extend(wasm::binary::CALL);
                self.cur_func.body.extend(0u32);
            }
            ast::Expr::If(if_expr) => todo!(),
            ast::Expr::Binary(binary_expr) => todo!(),
            ast::Expr::Unary(unary_expr) => todo!(),
            ast::Expr::Literal(literal) => match literal {
                ast::Literal::Bool(_) => todo!(),
                ast::Literal::Number(n) => {
                    self.gen_box(MemBox::F64, |state| {
                        state.cur_func.body.extend(wasm::binary::CONST_F64);
                        state.cur_func.body.extend(n);
                    });
                }
                ast::Literal::Str(_) => todo!(),
                ast::Literal::Nil => todo!(),
            },
            ast::Expr::Identifier(identifier) => todo!(),
        }
    }

    /// Boxes the top item on the stack and returns a pointer to it.
    ///
    /// `[T] -> [I32]`
    fn gen_box(&mut self, mem_box: MemBox, gen_value: impl Fn(&mut WasmGenState)) {
        let ptr = self.mem_store.alloc(mem_box.size());

        // Write the memory location
        self.cur_func.body.extend(wasm::binary::CONST_I32);
        self.cur_func.body.extend(ptr);

        // Generate the value to store
        gen_value(self);

        // Store that value in memory
        self.cur_func.body.extend(mem_box.instr_store());
        self.cur_func.body.extend([
            0x01u8, // Align 1
            0x00,   // Offset 0
        ]);

        // Return a pointer to the memory location
        self.cur_func.body.extend(wasm::binary::CONST_I32);
        self.cur_func.body.extend([ptr]);
    }

    /// Unboxes the pointer on top of the stack and returns the value.
    ///
    /// `[I32] -> [T]`
    fn gen_unbox(&mut self, mem_box: MemBox) {
        self.cur_func.body.extend(mem_box.instr_load());
        self.cur_func.body.extend([
            0x01u8, // Align 1
            0x00,   // Offset 0
        ]);
    }
}

struct MemStore {
    next_idx: u32,
}

impl MemStore {
    pub fn new() -> Self {
        MemStore { next_idx: 0 }
    }

    /// # Parameters:
    /// - `size`: The size in bytes.
    pub fn alloc(&mut self, size: u32) -> MemPtr {
        let idx = MemPtr(self.next_idx);
        self.next_idx += size;
        idx
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MemPtr(u32);
impl IntoBytes for MemPtr {
    fn into_bytes(self) -> Vec<u8> {
        self.0.into_bytes()
    }
}

/// The type of value to store in memory.
#[derive(Clone, Copy, Debug)]
enum MemBox {
    I32,
    F64,
}

impl MemBox {
    pub const fn size(self) -> u32 {
        match self {
            MemBox::I32 => 32 / 8,
            MemBox::F64 => 64 / 8,
        }
    }

    pub const fn instr_store(self) -> impl IntoBytes {
        match self {
            MemBox::I32 => wasm::binary::MEM_I32_STORE,
            MemBox::F64 => wasm::binary::MEM_F64_STORE,
        }
    }

    pub const fn instr_load(self) -> impl IntoBytes {
        match self {
            MemBox::I32 => wasm::binary::MEM_I32_LOAD,
            MemBox::F64 => wasm::binary::MEM_F64_LOAD,
        }
    }
}

mod wasm {
    use std::collections::HashMap;

    use binary::{IntoBytes, WasmVec};

    #[derive(Default)]
    pub struct Module {
        pub ty_sec: TypeSection,
        // No imports section here b/c it's better to put imports inside the
        // appropriate section (eg funcs section) to make indexing easier.
        pub funcs: Functions,
        pub mem_sec: MemorySection,
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
            if let Some(start_sec) = self.start_sec {
                buf.extend(start_sec.into_bytes());
            }
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

    pub struct MemIdx(u32);

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

    pub mod binary {
        use std::marker::PhantomData;

        pub const MAGIC_NUM: [u8; 4] = *b"\0asm";
        pub const VERSION: [u8; 4] = [0x01, 0x00, 0x00, 0x00];

        // Section types
        pub const SEC_CUSTOM: u8 = 0x00;
        pub const SEC_TY: u8 = 0x01;
        pub const SEC_IMPORT: u8 = 0x02;
        pub const SEC_FUNC: u8 = 0x03;
        pub const SEC_TABLE: u8 = 0x04;
        pub const SEC_MEM: u8 = 0x05;
        pub const SEC_GLOBAL: u8 = 0x06;
        pub const SEC_EXPORT: u8 = 0x07;
        pub const SEC_START: u8 = 0x08;
        pub const SEC_ELEM: u8 = 0x09;
        pub const SEC_CODE: u8 = 0x0A;
        pub const SEC_DATA: u8 = 0x0B;
        pub const SEC_DATA_COUNT: u8 = 0x0C;

        // Number types
        pub const TY_I32: u8 = 0x7F;
        pub const TY_I64: u8 = 0x7E;
        pub const TY_F32: u8 = 0x7D;
        pub const TY_F64: u8 = 0x7C;

        // Vector type
        pub const TY_VEC: u8 = 0x7B;

        // Reference types
        pub const TY_FUNC_REF: u8 = 0x70;
        pub const TY_EXTERN_REF: u8 = 0x6F;

        // Function type
        pub const TY_FUNC: u8 = 0x60;

        // Control instructions
        pub const CALL: u8 = 0x10;
        pub const END: u8 = 0x0B;

        // Variable instructions
        pub const LOCAL_GET: u8 = 0x20;
        pub const LOCAL_SET: u8 = 0x21;
        pub const LOCAL_TEE: u8 = 0x22;

        // Memory instructions
        pub const MEM_I32_LOAD: u8 = 0x28;
        pub const MEM_I32_STORE: u8 = 0x36;
        pub const MEM_F64_LOAD: u8 = 0x2B;
        pub const MEM_F64_STORE: u8 = 0x39;

        // Numeric
        pub const CONST_I32: u8 = 0x41;
        pub const CONST_F64: u8 = 0x44;

        /// Turn a section into bytecode with a proper header.
        ///
        /// See <https://webassembly.github.io/spec/core/binary/modules.html#sections>.
        pub fn sec_bytes(sec_id: u8, contents: impl IntoBytes) -> Vec<u8> {
            let contents = contents.into_bytes();

            let mut buf = vec![sec_id];

            let size = contents.len() as u32;
            buf.extend(size.into_bytes());

            buf.extend(contents);

            buf
        }

        pub trait IntoBytes {
            fn into_bytes(self) -> Vec<u8>;
        }

        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        pub struct WasmVec<T> {
            vec: Vec<u8>,
            size: u32,
            _phantom: PhantomData<T>,
        }

        impl<T> WasmVec<T> {
            pub fn new() -> Self {
                WasmVec {
                    vec: Vec::new(),
                    size: 0,
                    _phantom: PhantomData,
                }
            }

            pub fn extend<I: IntoIterator<Item = T>>(&mut self, items: I)
            where
                T: IntoBytes,
            {
                self.vec.extend(items.into_iter().flat_map(|i| {
                    self.size += 1;
                    i.into_bytes()
                }))
            }

            pub fn size(&self) -> u32 {
                self.size
            }
        }

        impl<T> IntoBytes for WasmVec<T> {
            fn into_bytes(self) -> Vec<u8> {
                let mut buf = Vec::new();

                // Write the header
                buf.extend(self.size.into_bytes());

                // Write the actual bytes
                buf.extend(self.vec);

                buf
            }
        }

        impl<T> Default for WasmVec<T> {
            fn default() -> Self {
                WasmVec::new()
            }
        }

        impl<T> std::ops::Deref for WasmVec<T> {
            type Target = Vec<u8>;

            fn deref(&self) -> &Self::Target {
                &self.vec
            }
        }

        impl<A: IntoBytes> FromIterator<A> for WasmVec<A> {
            fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
                let mut vec = WasmVec::new();
                vec.extend(iter);
                vec
            }
        }

        /// The raw bytes of the instructions. Shoud not include the END opcode.
        #[derive(Debug)]
        pub struct Expr {
            instructions: Vec<u8>,
        }

        impl Expr {
            pub fn new() -> Self {
                Expr {
                    instructions: Vec::new(),
                }
            }

            pub fn extend(&mut self, instr: impl IntoBytes) {
                self.instructions.extend(instr.into_bytes());
            }
        }

        impl IntoBytes for Expr {
            fn into_bytes(self) -> Vec<u8> {
                let mut buf = self.instructions;
                buf.push(END);
                buf
            }
        }

        impl IntoBytes for u32 {
            fn into_bytes(self) -> Vec<u8> {
                let mut buf = Vec::new();
                leb128::write::unsigned(&mut buf, self.into()).unwrap();
                buf
            }
        }

        impl IntoBytes for u8 {
            fn into_bytes(self) -> Vec<u8> {
                vec![self]
            }
        }

        impl IntoBytes for f64 {
            fn into_bytes(self) -> Vec<u8> {
                self.to_le_bytes().to_vec()
            }
        }

        impl<const N: usize, T: IntoBytes> IntoBytes for [T; N] {
            fn into_bytes(self) -> Vec<u8> {
                self.into_iter().flat_map(|e| e.into_bytes()).collect()
            }
        }
    }
}
