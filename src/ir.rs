use std::vec;

use crate::ast::{self, Binding, BindingMetadata};
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
    fn gen_stmt(&mut self, func: &mut Func, stmt: ast::Stmt) -> Option<GenErr> {
        match stmt {
            ast::Stmt::Expr(e) => self.gen_expr(func, e),
            ast::Stmt::Let(binding) => self.gen_binding(func, binding),
            _ => Some(GenErr::new(
                "CodeGen".to_string(),
                "Couldn't generate statement".to_string(),
            )),
        }
    }
    fn gen_binding(&mut self, func: &mut Func, binding: ast::Binding) -> Option<GenErr> {
        match binding.metadata {
            BindingMetadata::Var => {
                let slot = func.addVar(&binding);
                self.gen_expr(func, binding.value);
                func.addCommand(CommandOps::Store, None, Some(vec![slot as f64]));
            }
            BindingMetadata::Func {
                arguments,
                upvalues,
            } => {}
        }
        None
    }
    fn gen_expr(&mut self, func: &mut Func, expr: ast::Expr) -> Option<GenErr> {
        match expr {
            ast::Expr::Call(call) => self.gen_call(func, call),
            ast::Expr::Literal(literal) => self.gen_literal_expr(func, literal),
            ast::Expr::Binary(bin) => self.gen_binary_expr(func, *bin),
            ast::Expr::Unary(u) => self.gen_unary_expr(func, *u),
            ast::Expr::If(ifs) => self.gen_if_expr(func, *ifs),
            ast::Expr::Block(bk) => self.gen_block_expr(func, bk),
            _ => Some(GenErr::new(
                "CodeGen".to_string(),
                "Couldn't generate expression".to_string(),
            )),
        }
    }
    fn gen_block_expr(&mut self, func: &mut Func, expr: ast::Block) -> Option<GenErr> {
        func.scope.id.push(func.scope.nid);
        func.scope.nid+=1;
        for stmt in expr.stmts {
            self.gen_stmt(func, stmt);
        }
        if expr.return_expr.is_some() {
            self.gen_expr(func, *(expr.return_expr.unwrap()));
        };
        func.scope.nid-=1;
        func.scope.id.pop();
        None
    }
    fn gen_if_expr(&mut self, func: &mut Func, expr: ast::IfExpr) -> Option<GenErr> {
        self.gen_expr(func, expr.condition);
        func.addCommand(CommandOps::Jz, None, None);
        let mut cmd = func.lastcmd.clone();
        self.gen_expr(func, ast::Expr::Block(expr.then_block));
        func.blocks[cmd[0]].contents[cmd[1]].inline_params = Some(vec![func.blocks.len() as f64]);
        func.addCommand(CommandOps::Jmp, None, None);
        cmd = func.lastcmd.clone();
        match expr.else_block {
            Some(ast::ElseBlock::ElseIf(e)) => self.gen_if_expr(func, *e),
            Some(ast::ElseBlock::Else(b)) => self.gen_expr(func, ast::Expr::Block(b)),
            None => None,
        };
        func.blocks[cmd[0]].contents[cmd[1]].inline_params = Some(vec![func.blocks.len() as f64]);
        func.blocks.push(Block {
            contents: Vec::new(),
            id: func.blocks.len() as u16,
        });
        return None;
    }
    fn gen_unary_expr(&mut self, func: &mut Func, expr: ast::UnaryExpr) -> Option<GenErr> {
        self.gen_expr(func, expr.rhs);
        match expr.op {
            ast::UnaryOp::Negate => {
                let inline_params = vec![func.addConstant(Value::int64(-1))];
                func.addCommand(
                    CommandOps::Const,
                    Some(DataType::new(DataTypeRaw::Int64, false)),
                    Some(inline_params),
                );
                func.addCommand(
                    CommandOps::Mul,
                    Some(DataType::new(DataTypeRaw::Float64, false)),
                    None,
                );
            }
            ast::UnaryOp::Not => {
                func.addCommand(
                    CommandOps::Not,
                    Some(DataType::new(DataTypeRaw::Bool, false)),
                    None,
                );
            }
            _ => {
                return Some(GenErr::new(
                    "CodeGen".to_string(),
                    "Couldn't generate unary expression".to_string(),
                ))
            }
        };
        return None;
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
            ast::BinaryOp::And => func.addCommand(
                CommandOps::And,
                Some(DataType::new(DataTypeRaw::Bool, false)),
                None,
            ),
            ast::BinaryOp::Or => func.addCommand(
                CommandOps::Or,
                Some(DataType::new(DataTypeRaw::Bool, false)),
                None,
            ),
            ast::BinaryOp::Add => func.addCommand(
                CommandOps::Add,
                Some(DataType::new(DataTypeRaw::Float64, false)),
                None,
            ),
            ast::BinaryOp::Subtract => func.addCommand(
                CommandOps::Sub,
                Some(DataType::new(DataTypeRaw::Float64, false)),
                None,
            ),
            ast::BinaryOp::Multiply => func.addCommand(
                CommandOps::Mul,
                Some(DataType::new(DataTypeRaw::Float64, false)),
                None,
            ),
            ast::BinaryOp::Divide => func.addCommand(
                CommandOps::Div,
                Some(DataType::new(DataTypeRaw::Float64, false)),
                None,
            ),
            ast::BinaryOp::Eq => func.addCommand(
                CommandOps::Eq,
                Some(DataType::new(DataTypeRaw::Bool, false)),
                None,
            ),
            ast::BinaryOp::NotEq => func.addCommand(
                CommandOps::NotEq,
                Some(DataType::new(DataTypeRaw::Bool, false)),
                None,
            ),
            ast::BinaryOp::GreaterEq => func.addCommand(
                CommandOps::GreaterEq,
                Some(DataType::new(DataTypeRaw::Bool, false)),
                None,
            ),
            ast::BinaryOp::LessEq => func.addCommand(
                CommandOps::LessEq,
                Some(DataType::new(DataTypeRaw::Bool, false)),
                None,
            ),
            ast::BinaryOp::Greater => func.addCommand(
                CommandOps::Greater,
                Some(DataType::new(DataTypeRaw::Bool, false)),
                None,
            ),
            ast::BinaryOp::Less => func.addCommand(
                CommandOps::Less,
                Some(DataType::new(DataTypeRaw::Bool, false)),
                None,
            ),
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
        /*FILL THIS IN */
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
            scope: None,
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
            lastcmd: Vec::new(),
            scope: Scope{id:Vec::new(),nid:0}
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
                scope: None,
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
                scope: None,
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
    lastcmd: Vec<usize>,
    scope: Scope,
    /*Stack Frame Layout
    --Parameters--
    Value p1
    Value p2
    --Local Vars--
    Symbol xyz
    Symbol y
    --Temporaries--
    Number n1
    Number n2
    Number n3
    After completion, all but the parameters and a final temporary, the return value, are popped off the stack
    */
}
impl Func {
    fn addCommand(
        &mut self,
        op: CommandOps,
        data_type: Option<DataType>,
        inline_params: Option<Vec<f64>>,
    ) {
        let id = self.blocks.len() - 1;
        let cmd = Command {
            op: op,
            data_type: data_type,
            inline_params: inline_params,
        };
        self.blocks[id].contents.push(cmd);
        if op == CommandOps::Jz || op == CommandOps::Jmp {
            self.blocks.push(Block {
                contents: Vec::new(),
                id: (id + 1) as u16,
            });
        }
        self.lastcmd = vec![id, self.blocks[id].contents.len() - 1];
    }
    fn addConstant(&mut self, constant: Value) -> f64 {
        let id = self.constants.len() as u32;
        let dtype = constant.clone().val_to_type();
        self.constants.push(Symbol {
            dtype,
            name: None,
            id,
            upvalue: false,
            value: Some(constant),
            scope: None,
        });
        return id as f64;
    }
    fn addVar(&mut self, var: &Binding) -> usize {
        self.symbols.push(Symbol {
            dtype: DataType::new(var.data_type, false),
            name: Some(var.ident.name.clone()),
            id: self.symbols.len() as u32,
            upvalue: false,
            value: None,
            scope: Some(self.scope.clone()),
        });
        return self.symbols.len() - 1;
    }
    fn getVar(&mut self, var: String)-> &Symbol{
        let mut candidates: Vec<&Symbol>=Vec::new();
        let mut rankings: Vec<[u32; 2]>=Vec::new();
        for sym in &self.symbols{
            if sym.name==Some(var.clone()){
                candidates.push(&sym);
            }
        }
        for symidx in 0..candidates.len(){
            let mut idx=0;
            let mut score=0;
            let sym =candidates[symidx];
            for i in (sym.scope.as_ref().unwrap().id.iter()){
                if *i==self.scope.id[idx]{score+=1}
                idx+=1;
            }
            rankings.push([symidx as u32,score]);
        }
        rankings.sort_by(|a, b|b[1].cmp(&a[1]));
        return &self.symbols[rankings[0][0] as usize];
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
    scope: Option<Scope>,
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
impl Value {
    fn val_to_type(self) -> DataType {
        match self {
            Self::int64(i) => DataType::new(DataTypeRaw::Int64, false),
            Self::int32(i) => DataType::new(DataTypeRaw::Int32, false),
            Self::float32(i) => DataType::new(DataTypeRaw::Float32, false),
            Self::float64(i) => DataType::new(DataTypeRaw::Float64, false),
            Self::boolean(i) => DataType::new(DataTypeRaw::Bool, false),
            Self::string(i) => DataType::new(DataTypeRaw::String, false),
            Self::nil() => DataType::new(DataTypeRaw::Nil, false),
            _ => DataType::new(DataTypeRaw::Any, false),
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
    Jz,    //Jumps to a block if the top value on the stack is zero
    Jmp,   //Jumps to a block within a function
    Call,  //Switches functions, returns to prev
    Load,  //Copies the value of a local variable to the top of the stack
    Store, //Pops, then stores the top value of the stack into a local variable
    Not,
}
#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
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
    Any,
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
#[derive(Clone)]
struct Scope{
    id:Vec<u32>,
    nid:u32,
}
