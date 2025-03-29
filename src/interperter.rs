mod env;
mod stdlib;

pub use env::Env;
pub use stdlib::stub_stdlib;

use crate::{
    ast::{
        BinaryExpr, BinaryOp, Binding, BindingMetadata, Block, Call, ElseBlock, Expr, Identifier,
        IfExpr, Literal, Program, Stmt, UnaryExpr, UnaryOp,
    },
    lexer::Pos,
};

pub fn interpert(program: Program, env: &mut Env) -> Result<Value> {
    program.evaluate(env)
}

pub type Result<T> = std::result::Result<T, Error>;

trait Evaluate {
    fn evaluate(&self, env: &mut Env) -> Result<Value>;
}

impl Evaluate for Program {
    fn evaluate(&self, env: &mut Env) -> Result<Value> {
        for stmt in self {
            stmt.evaluate(env)?;
        }
        Ok(Value::Nil)
    }
}

impl Evaluate for Stmt {
    fn evaluate(&self, env: &mut Env) -> Result<Value> {
        match self {
            Stmt::Let(binding) => binding.evaluate(env),
            Stmt::Expr(expr) => expr.evaluate(env),
        }
    }
}

impl Evaluate for Binding {
    fn evaluate(&self, env: &mut Env) -> Result<Value> {
        let value = match &self.metadata {
            // Don't `.evaluate()` anything for a function
            BindingMetadata::Func {
                arguments,
                upvalues,
            } => {
                let arguments = arguments.clone();
                let upvalues = upvalues
                    .iter()
                    .map(|upvalue| env.resolve_upvalue(upvalue.clone()).clone())
                    .collect();
                let body = self.value.clone();
                Value::Func(Func::User(UserFunc {
                    name: self.ident.name.clone(),
                    arguments,
                    upvalues,
                    body,
                }))
            }
            // But do for a variable
            BindingMetadata::Var => self.value.evaluate(env)?,
        };
        env.define(value);
        Ok(Value::Nil)
    }
}

impl Evaluate for Expr {
    fn evaluate(&self, env: &mut Env) -> Result<Value> {
        match self {
            Expr::Block(block) => block.evaluate(env),
            Expr::Call(call) => call.evaluate(env),
            Expr::If(if_expr) => if_expr.evaluate(env),
            Expr::Binary(binary_expr) => binary_expr.evaluate(env),
            Expr::Unary(unary_expr) => unary_expr.evaluate(env),
            Expr::Literal(literal) => literal.evaluate(env),
            Expr::Identifier(identifier) => identifier.evaluate(env),
        }
    }
}

impl Evaluate for Block {
    fn evaluate(&self, env: &mut Env) -> Result<Value> {
        for stmt in &self.stmts {
            stmt.evaluate(env)?;
        }

        self.return_expr
            .as_ref()
            .map(|e| e.evaluate(env))
            .unwrap_or(Ok(Value::Nil))
    }
}

impl Evaluate for Call {
    // TODO: split into seperate for user and native?
    fn evaluate(&self, env: &mut Env) -> Result<Value> {
        let func = self.target.evaluate(env)?;
        let Value::Func(func) = func else {
            return Err(Error::new(ErrorKind::TypeError {
                expected: DiagnosticType::Func,
                actual: func.into(),
            })
            .pos(self.pos));
        };

        if let Func::User(func) = func.clone() {
            let incorrect_arity = func.arguments.len() > self.arguments.len();
            if incorrect_arity {
                return Err(Error::new(ErrorKind::IncorrectArity {
                    given: self.arguments.len(),
                    correct: func.arguments.len(),
                })
                .pos(self.pos));
            }
        }

        let arguments: Vec<Value> = self
            .arguments
            .iter()
            .map(|value| value.evaluate(env))
            .collect::<Result<_>>()?;

        // Don't tail call for native funcs b/c they handle args differently
        if self.is_tail_call && matches!(func, Func::User(_)) {
            env.tail_call(func, arguments);
            return Ok(Value::TailCall);
        }

        // Create new frame *after* evaluating arguments
        let mut env = env.new_frame(func.clone());

        // Actually define arguments
        for arg in arguments.clone() {
            env.define(arg);
        }

        loop {
            let ret = match env.func().clone() {
                Func::User(func) => func.body.evaluate(&mut env)?,
                // native funcs don't support tce
                Func::Native(func) => break func.call(arguments),
            };

            if let Value::TailCall = ret {
                continue;
            } else {
                break Ok(ret);
            }
        }
    }
}

impl Evaluate for IfExpr {
    fn evaluate(&self, env: &mut Env) -> Result<Value> {
        let condition = self.condition.evaluate(env)?;
        if condition.is_truthy() {
            self.then_block.evaluate(env)
        } else {
            match &self.else_block {
                Some(ElseBlock::Else(else_block)) => Ok(else_block.evaluate(env)?),
                Some(ElseBlock::ElseIf(if_expr)) => Ok(if_expr.evaluate(env)?),
                None => Ok(Value::Nil),
            }
        }
    }
}

impl Evaluate for BinaryExpr {
    fn evaluate(&self, env: &mut Env) -> Result<Value> {
        use Value::{Bool, Num, Str};

        let lhs = self.lhs.evaluate(env)?;
        // A closure so that it is lazy, for short-circuiting
        let mut rhs = || self.rhs.evaluate(env);
        Ok(match &self.op {
            BinaryOp::Or => {
                if lhs.is_truthy() {
                    lhs
                } else {
                    rhs()?
                }
            }
            BinaryOp::And => {
                if lhs.is_falsy() {
                    lhs
                } else {
                    rhs()?
                }
            }
            BinaryOp::NotEq => Bool(lhs != rhs()?),
            BinaryOp::Eq => Bool(lhs == rhs()?),
            BinaryOp::Greater => Bool(lhs.as_num()? > rhs()?.as_num()?),
            BinaryOp::GreaterEq => Bool(lhs.as_num()? >= rhs()?.as_num()?),
            BinaryOp::Less => Bool(lhs.as_num()? < rhs()?.as_num()?),
            BinaryOp::LessEq => Bool(lhs.as_num()? <= rhs()?.as_num()?),
            BinaryOp::Subtract => Num(lhs.as_num()? - rhs()?.as_num()?),
            BinaryOp::Add => match (lhs, rhs()?) {
                (Num(a), Num(b)) => Num(a + b),
                (Str(a), Num(b)) => Str(a + &b.to_string()),
                (Num(a), Str(b)) => Str(a.to_string() + &b),
                (Str(a), Str(b)) => Str(a + &b),

                // Errors:
                (Num(_), b) | (Str(_), b) => {
                    return Err(Error::new(ErrorKind::TypeError {
                        expected: DiagnosticType::Num,
                        actual: b.into(),
                    })
                    .pos(self.op_pos))
                }
                (a, _) => {
                    return Err(Error::new(ErrorKind::TypeError {
                        expected: DiagnosticType::Num,
                        actual: a.into(),
                    })
                    .pos(self.op_pos))
                }
            },
            BinaryOp::Divide => Num(lhs.as_num()? / rhs()?.as_num()?),
            BinaryOp::Multiply => Num(lhs.as_num()? * rhs()?.as_num()?),
        })
    }
}

impl Evaluate for UnaryExpr {
    fn evaluate(&self, env: &mut Env) -> Result<Value> {
        let rhs = self.rhs.evaluate(env)?;
        match self.op {
            UnaryOp::Not => Ok(Value::Bool(!rhs.is_truthy())),
            UnaryOp::Negate => Ok(Value::Num(-rhs.as_num()?)),
        }
    }
}

impl Evaluate for Literal {
    fn evaluate(&self, _env: &mut Env) -> Result<Value> {
        Ok(match self {
            Literal::Bool(b) => Value::Bool(*b),
            Literal::Number(n) => Value::Num(*n),
            Literal::Str(s) => Value::Str(s.clone()),
            Literal::Nil => Value::Nil,
        })
    }
}

impl Evaluate for Identifier {
    fn evaluate(&self, env: &mut Env) -> Result<Value> {
        Ok(env
            .get(self.location.expect("parser should have resolved variable"))
            .clone())
    }
}

#[derive(Clone, Debug)]
pub enum Value {
    Bool(bool),
    Num(f64),
    Str(String),
    Func(Func),
    List(Vec<Value>),
    Nil,

    /// Indicates that a tail call should be performed
    TailCall,
}

impl Value {
    fn is_truthy(&self) -> bool {
        !self.is_falsy()
    }
    fn is_falsy(&self) -> bool {
        matches!(self, Value::Bool(false) | Value::Nil)
    }

    fn as_num(&self) -> Result<f64> {
        match self {
            Self::Num(n) => Ok(*n),
            _ => Err(Error::new(ErrorKind::TypeError {
                expected: DiagnosticType::Num,
                actual: DiagnosticType::from(self),
            })),
        }
    }

    fn as_str(&self) -> Result<String> {
        match self {
            Self::Str(s) => Ok(s.clone()),
            _ => Err(Error::new(ErrorKind::TypeError {
                expected: DiagnosticType::Str,
                actual: DiagnosticType::from(self),
            })),
        }
    }

    fn as_list(&self) -> Result<Vec<Value>> {
        match self {
            Self::List(l) => Ok(l.clone()),
            _ => Err(Error::new(ErrorKind::TypeError {
                expected: DiagnosticType::List,
                actual: DiagnosticType::from(self),
            })),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bool(l), Self::Bool(r)) => l == r,
            (Self::Num(l), Self::Num(r)) => l == r,
            (Self::Str(l), Self::Str(r)) => l == r,
            (Self::Func(_), Self::Func(_)) => false,
            (Self::List(l), Self::List(r)) => l == r,
            (Self::Nil, Self::Nil) => true,
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Func {
    User(UserFunc),
    Native(&'static dyn NativeFunc),
}

#[derive(Clone, Debug)]
pub struct UserFunc {
    name: String,
    arguments: Vec<Identifier>,
    upvalues: Vec<Value>,
    body: Expr,
}

pub trait NativeFunc: std::fmt::Debug {
    fn call(&self, arguments: Vec<Value>) -> Result<Value>;
}

#[derive(Debug)]
pub struct Error {
    pub kind: ErrorKind,
    pub pos: Option<Pos>,
}

impl Error {
    pub fn new(kind: ErrorKind) -> Self {
        Self { kind, pos: None }
    }

    pub fn pos(mut self, pos: Pos) -> Self {
        self.pos = Some(pos);
        self
    }
}

#[expect(dead_code, reason = "Pretty error printing not implemented yet")]
#[derive(Debug)]
pub enum ErrorKind {
    TypeError {
        expected: DiagnosticType,
        actual: DiagnosticType,
    },
    IOError(std::io::Error),
    IncorrectArity {
        given: usize,
        correct: usize,
    },
}

#[derive(Debug)]
pub enum DiagnosticType {
    Bool,
    Num,
    Str,
    Func,
    List,
    Nil,
    TailCall,
}
impl From<&Value> for DiagnosticType {
    fn from(value: &Value) -> Self {
        match value {
            Value::Bool(_) => Self::Bool,
            Value::Num(_) => Self::Num,
            Value::Str(_) => Self::Str,
            Value::Func(_) => Self::Func,
            Value::List(_) => Self::List,
            Value::Nil => Self::Nil,
            Value::TailCall => Self::TailCall,
        }
    }
}
impl From<Value> for DiagnosticType {
    fn from(value: Value) -> Self {
        Self::from(&value)
    }
}
