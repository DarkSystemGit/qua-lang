use crate::ast::{self, Program};

pub fn gen_wasm(program: Program) -> Vec<u8> {
    let mut module = wasm::Module::default();

    // The idx is always 0, at least until wasm supports multiple memories
    let _ = module.mem_sec.insert(wasm::MemType { min: 0, max: None });

    for stmt in program {
        match stmt {
            ast::Stmt::Let(binding) => match binding.metadata {
                ast::BindingMetadata::Var => todo!(),
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

                    let ty = module.ty_sec.insert(ty);
                    let locals = vec![];
                    let body = vec![];

                    let func = wasm::Func { ty, locals, body };
                    module.funcs.insert(func);
                }
            },
            ast::Stmt::Expr(expr) => todo!(),
        }
    }

    module.into_bytes()
}

mod wasm {
    use std::collections::HashMap;

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
        pub locals: Vec<ValType>,
        /// The raw bytes of the instructions. Must include the END opcode.
        pub body: Vec<Instr>,
    }

    pub struct Instr(u8);

    #[derive(Default)]
    pub struct MemorySection {
        memories: Vec<MemType>,
    }

    impl MemorySection {
        pub fn insert(&mut self, mem: MemType) -> MemIdx {
            let idx = MemIdx(self.memories.len());
            self.memories.push(mem);
            idx
        }
    }

    pub struct MemIdx(usize);

    pub struct MemType {
        // The minimum size, as a multiple of the page size.
        pub min: u32,
        // The maximum size, as a multiple of the page size.
        pub max: Option<u32>,
    }
}
