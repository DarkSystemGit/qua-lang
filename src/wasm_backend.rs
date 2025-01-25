use wasm::Wasm;

use crate::ast::{
    BinaryExpr, Binding, Block, Call, Expr, Identifier, IfExpr, Literal, Program, Stmt, UnaryExpr,
};

mod wasm {
    pub struct Wasm(Vec<u8>);
    impl Wasm {
        pub fn new() -> Self {
            #[rustfmt::skip]
            let empty = vec![
                // module header
                0x00, 0x61, 0x73, 0x6d,

                // wasm version
                0x01, 0x00, 0x00, 0x00,
            ];

            Wasm(empty)
        }

        pub fn into_bytes(self) -> Vec<u8> {
            self.0
        }
    }

    pub enum Opcode {}
}

pub fn gen_wasm(program: Program) -> Wasm {
    let mut out = Wasm::new();

    for stmt in program {
        todo!()
    }

    out
}
