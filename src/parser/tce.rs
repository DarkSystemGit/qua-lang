use crate::ast::{
    BinaryExpr, Binding, BindingMetadata, Block, Call, ElseBlock, Expr, IfExpr, Program, Stmt,
    UnaryExpr,
};

pub fn mark_tail_calls(program: &mut Program) {
    mark_functions(program)
}

// Better named than `mark_tail_calls`
fn mark_functions(stmts: &mut [Stmt]) {
    for stmt in stmts {
        let Stmt::Let(binding) = stmt else { return };
        mark_binding(binding);
    }
}

fn mark_binding(binding: &mut Binding) {
    // Select only functions
    let is_func = matches!(binding.metadata, BindingMetadata::Func { .. });
    if !is_func {
        return;
    }

    mark_expr(&mut binding.value, true);
}

/// Parameters:
///  - `is_final`: Whether or not this is the final expression in the function.
///    For example, the condition in an if statement is not final. However, it
///    could still contain a function definition that needs to be scanned.
fn mark_expr(expr: &mut Expr, is_final: bool) {
    match expr {
        Expr::Block(block) => mark_block(block, is_final),
        Expr::Call(call) if is_final => call.is_tail_call = true,
        Expr::If(if_expr) => mark_if_expr(if_expr, is_final),
        Expr::Binary(binary_expr) => {
            mark_expr(&mut binary_expr.lhs, false);
            mark_expr(&mut binary_expr.rhs, false);
        }
        Expr::Unary(unary_expr) => mark_expr(&mut unary_expr.rhs, false),
        Expr::Literal(_) | Expr::Identifier(_) | Expr::Call(_) => {}
    }
}

fn mark_block(block: &mut Block, is_final: bool) {
    mark_functions(&mut block.stmts);
    if let Some(expr) = &mut block.return_expr {
        mark_expr(expr, is_final);
    }
}

fn mark_if_expr(if_expr: &mut Box<IfExpr>, is_final: bool) {
    mark_expr(&mut if_expr.condition, false);
    mark_block(&mut if_expr.then_block, is_final);
    match &mut if_expr.else_block {
        Some(ElseBlock::ElseIf(if_expr)) => mark_if_expr(if_expr, is_final),
        Some(ElseBlock::Else(block)) => mark_block(block, is_final),
        None => {}
    }
}
