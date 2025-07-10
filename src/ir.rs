use crate::ast;
use structx::*;
struct CodeGen {
    modules: Vec<Module>,
}
impl CodeGen {
    fn new(&mut self) {
        self.modules = Vec::new();
        let m = Module::new(0);
        self.modules.push(m);
    }
    fn gen_fn(
        &mut self,
        name: String,
        args: Vec<Symbol>,
        return_type: DataType,
        parent: u8,
        exported: bool,
        imported_by: Vec<u8>,
    ) -> &Func {
        for m in imported_by {
            self.modules[m as usize].addFn(
                name.clone(),
                args.clone(),
                return_type.clone(),
                false,
                true,
            );
        }
        let fnid = self.modules[parent as usize].addFn(
            name.clone(),
            args.clone(),
            return_type.clone(),
            exported,
            false,
        );
        return &self.modules[parent as usize].functions[fnid];
    }
    fn gen_expr(&mut self, func: &mut Func, expr: ast::Expr) -> Option<GenErr> {
        match expr {
            ast::Expr::Call(call) => self.gen_call(func, call),
            ast::Expr::Literal(literal) => self.gen_literal_expr(func, literal),
            ast::Expr::Binary(bin)=>self.gen_binary_expr(func, * bin),
            _ => Some(GenErr::new(
                "CodeGen".to_string(),
                "Couldn't generate expression".to_string(),
            )),
        }
    }
    fn gen_literal_expr(&mut self, func: &mut Func, expr: ast::Literal) -> Option<GenErr> {
        match expr {
            ast::Literal::Bool(b) => {
                let inline_params = vec![func.addConstant(Value::boolean(b))];
                func.addCommand(
                    CommandOps::Const,
                    Some(DataType::new(DataTypeRaw::Bool, false)),
                    Some(inline_params),
                );
            }
            ast::Literal::Nil => {
                let inline_params = vec![func.addConstant(Value::nil())];
                func.addCommand(
                    CommandOps::Const,
                    Some(DataType::new(DataTypeRaw::Nil, false)),
                    Some(inline_params),
                );
            }
            ast::Literal::Number(n) => {
                if (n.fract() == 0.0) {
                     let inline_params = vec![func.addConstant(Value::int64(n as i64))];
                    func.addCommand(
                        CommandOps::Const,
                        Some(DataType::new(DataTypeRaw::Int64, false)),
                        Some(inline_params),
                    );
                } else {
                    let inline_params = vec![func.addConstant(Value::float64(n))];
                    func.addCommand(
                        CommandOps::Const,
                        Some(DataType::new(DataTypeRaw::Float64, false)),
                        Some(inline_params),
                    );
                }
            }
            ast::Literal::Str(s) => {
                let inline_params = vec![func.addConstant(Value::string(s))];
                func.addCommand(
                    CommandOps::Const,
                    Some(DataType::new(DataTypeRaw::String, false)),
                    Some(inline_params),
                );
            }
            _ => {
                return Some(GenErr::new(
                    "CodeGen".to_string(),
                    "Couldn't generate literal".to_string(),
                ))
            }
        };
        return None;
    }
    fn gen_binary_expr(&mut self, func: &mut Func, expr: ast::BinaryExpr) -> Option<GenErr> {
        self.gen_expr(func, expr.rhs);
        self.gen_expr(func, expr.lhs);
        match expr.op {
            ast::BinaryOp::And => func.addCommand(CommandOps::And, Some(DataType::new(DataTypeRaw::Bool, false)), None),
            ast::BinaryOp::Or => func.addCommand(CommandOps::Or, Some(DataType::new(DataTypeRaw::Bool, false)), None),
            ast::BinaryOp::Add => func.addCommand(CommandOps::Add, Some(DataType::new(DataTypeRaw::Float64, false)), None),
            ast::BinaryOp::Subtract => func.addCommand(CommandOps::Sub, Some(DataType::new(DataTypeRaw::Float64, false)), None),
            ast::BinaryOp::Multiply => func.addCommand(CommandOps::Mul, Some(DataType::new(DataTypeRaw::Float64, false)), None),
            ast::BinaryOp::Divide => func.addCommand(CommandOps::Div, Some(DataType::new(DataTypeRaw::Float64, false)), None),
            ast::BinaryOp::Eq => func.addCommand(CommandOps::Eq, Some(DataType::new(DataTypeRaw::Bool, false)), None),
            ast::BinaryOp::NotEq => func.addCommand(CommandOps::NotEq, Some(DataType::new(DataTypeRaw::Bool, false)), None),
            ast::BinaryOp::GreaterEq => func.addCommand(CommandOps::GreaterEq, Some(DataType::new(DataTypeRaw::Bool, false)), None),
            ast::BinaryOp::LessEq => func.addCommand(CommandOps::LessEq, Some(DataType::new(DataTypeRaw::Bool, false)), None),
            ast::BinaryOp::Greater => func.addCommand(CommandOps::Greater, Some(DataType::new(DataTypeRaw::Bool, false)), None),
            ast::BinaryOp::Less => func.addCommand(CommandOps::Less, Some(DataType::new(DataTypeRaw::Bool, false)), None),
            _ => {
                return Some(GenErr::new(
                    "CodeGen".to_string(),
                    "Couldn't generate binary expression".to_string(),
                ))
            }
        }
        return None;
    }
    fn resolve_fn(&self, call: &ast::Expr) -> &Func {
        return;
    }
    fn gen_call(&mut self, parent_func: &mut Func, call: ast::Call) -> Option<GenErr> {
        for arg in call.arguments {
            self.gen_expr(parent_func, arg);
        }
        let target = self.resolve_fn(&call.target);
        parent_func.addCommand(
            CommandOps::Call,
            Some(target.return_type.clone()),
            Some(vec![target.id.into()]),
        );
        return None;
    }
}
struct GenErr {
    errtype: String,
    msg: String,
}
impl GenErr {
    fn new(errtype: String, msg: String) -> Self {
        GenErr { errtype, msg }
    }
    fn print(&self) {
        eprintln!("[{} Error] {}", self.errtype, self.msg);
    }
}
struct Module {
    globals: Vec<Symbol>,
    functions: Vec<Func>,
    exports: Vec<Symbol>,
    imports: Vec<Symbol>,
    id: u16,
}
impl Module {
    fn new(id: u16) -> Self {
        Module {
            globals: Vec::new(),
            functions: Vec::new(),
            exports: Vec::new(),
            imports: Vec::new(),
            id: id,
        }
    }
    fn addGlobal(&mut self, name: String, dtype: DataType, upvalue: bool) {
        let id = self.globals.len();
        self.globals.push(Symbol {
            dtype: dtype,
            name: Some(name),
            id: id as u32,
            upvalue: upvalue,
            value: Option::None,
        });
    }
    fn addFn(
        &mut self,
        name: String,
        args: Vec<Symbol>,
        return_type: DataType,
        exported: bool,
        imported: bool,
    ) -> usize {
        let mut func = Func {
            id: self.functions.len() as u16,
            name: name.clone(),
            parameters: args,
            return_type: return_type,
            symbols: Vec::new(),
            blocks: Vec::new(),
            constants: Vec::new(),
        };
        let id = func.id;
        func.blocks.push(Block {
            contents: Vec::new(),
            id: 0,
        });
        let fnref = self.functions.len();
        self.functions.push(func);
        if exported {
            self.exports.push(Symbol {
                dtype: DataType {
                    raw: DataTypeRaw::Fn,
                    isptr: true,
                    constid: Option::None,
                },
                name: Some(name.clone()),
                id: id as u32,
                upvalue: false,
                value: Option::None,
            })
        } else if imported {
            self.imports.push(Symbol {
                dtype: DataType {
                    raw: DataTypeRaw::Fn,
                    isptr: true,
                    constid: Option::None,
                },
                name: Some(name.clone()),
                id: id as u32,
                upvalue: false,
                value: Option::None,
            })
        }
        return fnref;
    }
}
struct Func {
    symbols: Vec<Symbol>,
    blocks: Vec<Block>,
    name: String,
    parameters: Vec<Symbol>,
    return_type: DataType,
    constants: Vec<Symbol>,
    id: u16,
}
impl Func {
    fn addCommand(
        &mut self,
        op: CommandOps,
        data_type: Option<DataType>,
        inline_params: Option<Vec<f64>>,
    ) {
        let id = self.blocks.len();
        let cmd = Command {
            op: op,
            data_type: data_type,
            inline_params: inline_params,
        };
        self.blocks[id].contents.push(cmd);
        if op == CommandOps::Jz {
            self.blocks.push(Block {
                contents: Vec::new(),
                id: id as u16,
            });
        }
    }
    fn addConstant(&mut self, constant: Value) -> f64 {
        let id = self.constants.len() as u32;
        let dtype=constant.clone().val_to_type();
        self.constants.push(Symbol {
            dtype,
            name: None,
            id,
            upvalue: false,
            value: Some(constant),
        });
        return id as f64;
    }
}
struct Block {
    contents: Vec<Command>,
    id: u16,
}
#[derive(Clone)]
struct Symbol {
    dtype: DataType,
    name: Option<String>,
    id: u32,
    upvalue: bool,
    value: Option<Value>,
}
#[derive(Clone)]
enum Value {
    int64(i64),
    float64(f64),
    int32(i32),
    float32(f32),
    boolean(bool),
    nil(),
    string(String),
}
impl Value{
    fn val_to_type(self)->DataType{
        match self{
            Self::int64(i)=>DataType::new(DataTypeRaw::Int64, false),
            Self::int32(i)=>DataType::new(DataTypeRaw::Int32, false),
            Self::float32(i)=>DataType::new(DataTypeRaw::Float32, false),
            Self::float64(i)=>DataType::new(DataTypeRaw::Float64, false),
            Self::boolean(i)=>DataType::new(DataTypeRaw::Bool, false),
            Self::string(i)=>DataType::new(DataTypeRaw::String, false),
            Self::nil()=>DataType::new(DataTypeRaw::Nil, false),
            _=>DataType::new(DataTypeRaw::Any, false),
            
        }
    }
}
struct Command {
    op: CommandOps,
    data_type: Option<DataType>, //for numeric ops, either Float(32/64) or Int(32/64)
    inline_params: Option<Vec<f64>>, //constant insertion
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
    Greater,
    GreaterEq,
    Less,
    LessEq,
    Eq,
    And,
    NotEq,
    Mod,
    Pop,
    Jz,
    Call,
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DataTypeRaw {
    Int32,
    Int64,
    Float32,
    Float64,
    Bool,
    Fn,
    String,
    Void,
    Nil,
    Any
}
#[derive(Clone)]
pub struct DataType {
    raw: DataTypeRaw,
    isptr: bool,
    constid: Option<u16>,
}
impl DataType {
    fn new(dr: DataTypeRaw, isptr: bool) -> Self {
        DataType {
            raw: dr,
            isptr,
            constid: None,
        }
    }
}
