use crate::ast::{self, Program};

pub fn gen_wasm(program: Program) -> Vec<u8> {
    let mut module = wasm::Module::default();

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

                    let ty_idx = module.ty_sec.insert(ty);
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
}
