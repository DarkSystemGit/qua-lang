use crate::ast;
use structx::*;
struct Module{
    globals: Vec<Symbol>,
    functions: Vec<Func>,
    exports:Vec<Symbol>,
    imports:Vec<Symbol>,
    id: u16,
}
struct Block{
    contents: Vec<Command>,
    id: u16,
}
struct Func{
    symbols: Vec<Symbol>,
    body: Vec<Block>,
    name: String,
    parameters: Vec<Symbol>,
    return_type:DataType,
    id: u16,
}
struct Symbol{
    dtype: DataType, 
    name: String,
    id: u32,
    upvalue: bool
}
struct Command{
    op: Commands,
    data_type: DataType,
    inline_params: Vec<f64>
}
enum Commands{
    Add,
    Sub,
    Mul,
    Div,
    Const,
    Load,
    Xor,
    Or,
    And,
    Not,
    Mod,
    Pop,
}
enum DataTypeRaw{
    Int32,
    Int64,
    Float32,
    Float64,
    Bool
}
struct DataType{
    raw: DataTypeRaw,
    isptr: bool,
    constid: u16,
}
