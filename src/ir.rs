use crate::ast;
use structx::*;

struct Module {
    globals: Vec<Symbol>,
    functions: Vec<Func>,
    exports: Vec<Symbol>,
    imports: Vec<Symbol>,
    id: u16,
}
impl Module {
    fn new(&mut self, id: u16){
        self.id=id;
    }
    
    fn addFn(&mut self, name: String, args: Vec<Symbol>, return_type: DataType, exported: bool, imported: bool) {
        let mut func=Func {
            id: self.functions.len() as u16,
            name: name.clone(),
            parameters: args,
            return_type: return_type,
            symbols: Vec::new(),
            blocks: Vec::new(),
        };
        let id=func.id;
        func.blocks.push(Block { contents: Vec::new(), id: 0 });
        self.functions.push(func);
        if exported {
            self.exports.push(Symbol {
                dtype: DataType {
                    raw: DataTypeRaw::Fn,
                    isptr: true,
                    constid: Option::None,
                },
                name: name.clone(),
                id: id as u32,
                upvalue: false,
            })
        }else if imported {
            self.imports.push(Symbol {
                dtype: DataType {
                    raw: DataTypeRaw::Fn,
                    isptr: true,
                    constid: Option::None,
                },
                name: name.clone(),
                id: id as u32,
                upvalue: false,
            })
        }
    }
}
struct Func {
    symbols: Vec<Symbol>,
    blocks: Vec<Block>,
    name: String,
    parameters: Vec<Symbol>,
    return_type: DataType,
    id: u16,
}
impl Func {
    fn addCommand(&mut self,op: CommandOps,data_type: Option<DataType>,inline_params: Option<Vec<f64>>){
        let id=self.blocks.len()-1;
        let cmd=Command { op: op, data_type: data_type, inline_params: inline_params }
        if op!=CommandOps::Jz{
            self.blocks[id].contents.push(cmd);
        }else{
            self.blocks.push(Block { contents: Vec::new(), id: id as u16});
            self.blocks[id+1].contents.push(cmd);
        }
    }
}
struct Block {
    contents: Vec<Command>,
    id: u16,
}
struct Symbol {
    dtype: DataType,
    name: String,
    id: u32,
    upvalue: bool,
}
struct Command {
    op: CommandOps,
    data_type: Option<DataType>, //for numeric ops, either Float(32/64) or Int(32/64)
    inline_params: Option<Vec<f64>>,//constant insertion
}
#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(u8)]
enum CommandOps {
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
    Jz,
}
enum DataTypeRaw {
    Int32,
    Int64,
    Float32,
    Float64,
    Bool,
    Fn,
}
struct DataType {
    raw: DataTypeRaw,
    isptr: bool,
    constid: Option<u16>,
}
