use wasm::binary::IntoBytes;

use crate::ast;

pub fn gen_wasm(program: ast::Program) -> Vec<u8> {
    let mut state = WasmGenState::new();
    state.gen_program(program);
    state.module.into_bytes()
}

struct WasmGenState {
    module: wasm::Module,
    cur_func: wasm::Func,
    mem_store: MemStore,
}
impl WasmGenState {
    fn new() -> Self {
        let mut module = wasm::Module::default();

        // The idx is always 0, at least until wasm supports multiple memories
        let _ = module.mem_sec.insert(wasm::MemType {
            limits: wasm::Limits { min: 0, max: None },
        });

        let main_func = {
            let ty = wasm::FuncType {
                params: vec![],
                results: vec![],
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
                self.gen_box();

                let idx = self.cur_func.insert_local(wasm::ValType::I32);
                self.cur_func.body.extend([wasm::binary::LOCAL_SET]);
                self.cur_func.body.extend([idx]);
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
                    results: vec![result_ty],
                };

                let ty = self.module.ty_sec.insert(ty);
                let func = wasm::Func::new(ty);

                // TODO: actual generate body

                self.module.funcs.insert(func);
            }
        }
    }

    fn gen_expr(&mut self, expr: ast::Expr) {
        todo!()
    }

    /// Boxes the top item on the stack and returns a pointer to it.
    ///
    /// `[I32] -> [I32]`
    fn gen_box(&mut self) {
        let idx = self.mem_store.alloc(4);

        // Write the memory location
        self.cur_func.body.extend([idx]);

        // Store that value in memory
        self.cur_func.body.extend([
            wasm::binary::MEM_I32_STORE,
            // Align 0
            0x01,
            // Offset 0
            0x00,
        ]);

        // Return a pointer to the memory location
        self.cur_func.body.extend([idx]);
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
    pub fn alloc(&mut self, size: u32) -> MemIdx {
        let idx = MemIdx(self.next_idx);
        self.next_idx += size;
        idx
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MemIdx(u32);
impl IntoBytes for MemIdx {
    fn into_bytes(self) -> Vec<u8> {
        self.0.into_bytes()
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
    }

    impl Module {
        pub fn into_bytes(self) -> Vec<u8> {
            todo!()
        }
    }

    #[derive(Default)]
    pub struct TypeSection {
        types_map: HashMap<FuncType, TypeIdx>,
        types: Vec<FuncType>,
    }

    impl TypeSection {
        pub fn insert(&mut self, ty: FuncType) -> TypeIdx {
            if let Some(idx) = self.types_map.get(&ty) {
                *idx
            } else {
                let next_idx = self.types.len();
                self.types.push(ty.clone());

                let idx = TypeIdx(next_idx);
                self.types_map.insert(ty, idx);

                idx
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub struct TypeIdx(usize);

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct FuncType {
        pub params: Vec<ValType>,
        pub results: Vec<ValType>,
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

    /// *Not* the functions section. It is a merge of the function, code, and
    /// (function) imports section.
    #[derive(Default)]
    pub struct Functions {
        imports: Vec<FuncImport>,
        funcs: Vec<Func>,
    }

    impl Functions {
        fn next_idx(&self) -> FuncIdx {
            FuncIdx(self.imports.len() + self.funcs.len())
        }

        pub fn insert(&mut self, func: Func) -> FuncIdx {
            let idx = self.next_idx();
            self.funcs.push(func);
            idx
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub struct FuncIdx(usize);

    pub struct FuncImport {
        pub module: String,
        pub name: String,
        pub ty: TypeIdx,
    }

    pub struct Func {
        pub ty: TypeIdx,
        locals: WasmVec<ValType>,
        /// The raw bytes of the instructions. Shoud not include the END opcode.
        pub body: WasmVec<Instr>,
    }

    impl Func {
        pub fn new(ty: TypeIdx) -> Self {
            Func {
                ty,
                locals: WasmVec::new(),
                body: WasmVec::new(),
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

    pub type Instr = u8;

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

    pub struct MemIdx(u32);

    pub struct MemType {
        // The min and max size, in multiples of the page size.
        pub limits: Limits,
    }

    impl IntoBytes for MemType {
        fn into_bytes(self) -> Vec<u8> {
            todo!()
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

    pub mod binary {
        use std::marker::PhantomData;

        pub const MAGIC_NUM: [u8; 4] = *b"\0asm";
        pub const VERSION: [u8; 4] = [0x01, 0x00, 0x00, 0x00];

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
        pub const END: u8 = 0x0B;

        // Variable instructions
        pub const LOCAL_GET: u8 = 0x20;
        pub const LOCAL_SET: u8 = 0x21;
        pub const LOCAL_TEE: u8 = 0x22;

        // Memory instructions
        pub const MEM_I32_LOAD: u8 = 0x28;
        pub const MEM_I32_STORE: u8 = 0x36;

        // Numeric
        pub const CONST_I32: u8 = 0x41;

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

            pub fn extend<V: IntoBytes, I: IntoIterator<Item = V>>(&mut self, items: I) {
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
    }
}
