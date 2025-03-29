use crate::lexer::Pos;

pub type Program = Vec<Stmt>;

#[derive(Clone, Debug)]
pub enum Stmt {
    Let(Binding),
    Expr(Expr),
}

#[derive(Clone, Debug)]
pub struct Binding {
    pub ident: Identifier,
    pub metadata: BindingMetadata,
    pub value: Expr,
}

#[derive(Clone, Debug)]
pub enum BindingMetadata {
    Var,
    Func {
        arguments: Vec<Identifier>,
        upvalues: Vec<Upvalue>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Upvalue {
    pub target: IdentLocation,
    pub dbg_name: String,
}

#[derive(Clone, Debug)]
pub enum Expr {
    Block(Block),
    Call(Call),
    If(Box<IfExpr>),
    Binary(Box<BinaryExpr>),
    Unary(Box<UnaryExpr>),
    Literal(Literal),
    Identifier(Identifier),
}

#[derive(Clone, Debug)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub return_expr: Option<Box<Expr>>,
}

#[derive(Clone, Debug)]
pub struct Call {
    pub target: Box<Expr>,
    pub arguments: Vec<Expr>,
    pub is_tail_call: bool,
    pub pos: Pos,
}

#[derive(Clone, Debug)]
pub struct IfExpr {
    pub condition: Expr,
    pub then_block: Block,
    pub else_block: Option<ElseBlock>,
}
#[derive(Clone, Debug)]
pub enum ElseBlock {
    ElseIf(Box<IfExpr>),
    Else(Block),
}

#[derive(Clone, Debug)]
pub struct BinaryExpr {
    pub lhs: Expr,
    pub op: BinaryOp,
    pub rhs: Expr,

    pub op_pos: Pos,
}
#[derive(Clone, Debug)]
pub enum BinaryOp {
    Or,
    And,

    NotEq,
    Eq,

    Greater,
    GreaterEq,
    Less,
    LessEq,

    Subtract,
    Add,

    Divide,
    Multiply,
}

#[derive(Clone, Debug)]
pub struct UnaryExpr {
    pub op: UnaryOp,
    pub rhs: Expr,
}
#[derive(Clone, Debug)]
pub enum UnaryOp {
    Not,
    Negate,
}

#[derive(Clone, Debug)]
pub enum Literal {
    Bool(bool),
    Number(f64),
    Str(String),
    Nil,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Identifier {
    pub name: String,
    pub location: Option<IdentLocation>,
}
impl Identifier {
    pub fn new(name: String) -> Self {
        Self {
            name,
            location: None,
        }
    }

    pub fn resolve(mut self, location: IdentLocation) -> Self {
        self.location = Some(location);
        self
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum IdentLocation {
    Stack(StackIndex),
    Upvalue(UpvalueIndex),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct StackIndex(pub usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct UpvalueIndex(pub usize);
