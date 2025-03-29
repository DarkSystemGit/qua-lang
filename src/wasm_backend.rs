use crate::ast::Program;

pub fn gen_wasm(program: Program) -> Module {
    let mut module = Module::new();

    todo!()
}

pub struct Module {
    function_types: Vec<FuncType>,
    functions: Vec<Func>,
}

impl Module {
    pub fn new() -> Self {
        Module {
            function_types: Vec::new(),
            functions: Vec::new(),
        }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        todo!()
    }
}

struct FuncTypeIdx(usize);
struct FuncType {
    parameters: Vec<ValueType>,
    results: Vec<ValueType>,
}

enum ValueType {
    Num(NumType),
}

enum NumType {
    I32,
    I64,
    F32,
    F64,
}

struct Func {
    ty: FuncTypeIdx,
    locals: Vec<ValueType>,
    body: Vec<Instr>,
}

enum Instr {
    I32Const(i32),
    I64Const(i64),
    F32Const(f32),
    F64Const(f64),
}
