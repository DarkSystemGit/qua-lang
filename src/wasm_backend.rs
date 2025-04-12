use wasm::binary::{IntoBytes, WasmVec};

use crate::ast;

mod wasm;

pub fn gen_wasm(program: ast::Program) -> Vec<u8> {
    WasmGenState::gen(program)
}

struct WasmGenState {
    module: wasm::Module,
    mem_store: MemStore,
}
impl WasmGenState {
    fn gen(program: ast::Program) -> Vec<u8> {
        let mut module = wasm::Module::default();

        let import_host_print = wasm::FuncImport {
            module: wasm::Name("host".to_string()),
            name: wasm::Name("print".to_string()),
            ty: {
                let ty = wasm::FuncType {
                    params: [MEM_PTR_TY].into_iter().collect(),
                    results: WasmVec::new(),
                };
                module.ty_sec.insert(ty)
            },
        };
        module.funcs.insert_import(import_host_print);

        let mut table_sec = wasm::TableSection::new();
        table_sec.insert(wasm::TableType {
            limits: wasm::Limits { min: 64, max: None },
            ty: wasm::RefType::Func,
        });
        module.table_sec = Some(table_sec);

        let mem_idx = module.mem_sec.insert(wasm::MemType {
            limits: wasm::Limits { min: 64, max: None },
        });

        let mut main_func = {
            let ty = wasm::FuncType {
                params: WasmVec::new(),
                results: WasmVec::new(),
            };
            let ty = module.ty_sec.insert(ty);
            wasm::Func::new_no_implicit_self_ref(ty, 0)
        };

        let mut state = WasmGenState {
            module,
            mem_store: MemStore::new(mem_idx),
        };
        state.gen_program(&mut main_func, program);
        state.finish(main_func)
    }

    fn finish(mut self, func: wasm::Func) -> Vec<u8> {
        // Set up main function
        let main_idx = self.module.funcs.insert(func);

        // let start_sec = wasm::StartSection { func: idx };
        // self.module.start_sec = Some(start_sec);
        let mut export_sec = wasm::ExportSection::new();
        export_sec.insert(wasm::Export {
            name: wasm::Name("main".to_string()),
            desc: wasm::ExportDesc::Func(main_idx),
        });
        export_sec.insert(wasm::Export {
            name: wasm::Name("mem".to_string()),
            desc: wasm::ExportDesc::Mem(self.mem_store.mem_idx),
        });
        self.module.export_sec = Some(export_sec);

        // Populate elem section
        let mut elem_sec = wasm::ElemSection::new();
        let mut elem_segment = wasm::Elem::default();
        elem_segment.insert(self.module.funcs.all_idxs());
        elem_sec.insert(elem_segment);
        self.module.elem_sec = Some(elem_sec);

        self.module.into_bytes()
    }

    fn gen_program(&mut self, func: &mut wasm::Func, program: ast::Program) {
        for stmt in program {
            self.gen_stmt(func, stmt);
        }
    }

    fn gen_stmt(&mut self, func: &mut wasm::Func, stmt: ast::Stmt) {
        match stmt {
            ast::Stmt::Let(binding) => self.gen_binding(func, binding),
            ast::Stmt::Expr(expr) => {
                self.gen_expr(func, expr);

                // TODO: *all* expressions must generate some return value,
                //       even if it's nil. Therefore, they must all be dropped.
            }
        }
    }

    fn gen_binding(&mut self, func: &mut wasm::Func, binding: ast::Binding) {
        match binding.metadata {
            ast::BindingMetadata::Var => {
                self.gen_expr(func, binding.value);
            }
            ast::BindingMetadata::Func {
                arguments,
                upvalues,
            } => {
                let ty = wasm::FuncType::new(arguments.len(), MEM_PTR_TY);
                let ty = self.module.ty_sec.insert(ty);
                let mut new_func = wasm::Func::new(ty, arguments.len() as u32);

                self.gen_expr(&mut new_func, binding.value);

                let idx = self.module.funcs.insert(new_func);
                let idx = self.module.funcs.raw_func_sec_idx(idx);
                func.gen_box(
                    self.mem_store.alloc(wasm::BoxType::Func),
                    [|func: &mut wasm::Func| {
                        func.body.extend(wasm::binary::CONST_I32);
                        func.body.extend(idx);
                    }],
                );
            }
        }

        func.gen_local_set(
            MEM_PTR_TY,
            Some(binding.ident.location.expect(&format!(
                "location resolved for ident, {}",
                binding.ident.name
            ))),
        );
    }

    fn gen_expr(&mut self, func: &mut wasm::Func, expr: ast::Expr) {
        // TODO: all expressions must return some value, even if it is nil.
        match expr {
            ast::Expr::Block(block) => {
                for stmt in block.stmts {
                    self.gen_stmt(func, stmt);
                }

                if let Some(return_expr) = block.return_expr {
                    self.gen_expr(func, *return_expr);
                }
            }
            ast::Expr::Call(call) => {
                match *call.target {
                    ast::Expr::Identifier(identifier) if identifier.name == "print" => {
                        for arg in call.arguments {
                            self.gen_expr(func, arg);
                        }

                        func.body.extend(wasm::binary::CALL);
                        func.body.extend(0u32);
                    }
                    expr => {
                        // Put arguments onto the stack
                        // Start with self reference
                        self.gen_expr(func, expr);
                        let target_idx = func.gen_local_tee(MEM_PTR_TY, None);
                        // Then the real args
                        // Save this before call.arguments is consumed in the for loop
                        let num_args = call.arguments.len();
                        for arg in call.arguments {
                            self.gen_expr(func, arg);
                        }

                        func.gen_local_get(target_idx);
                        func.gen_unbox(wasm::BoxType::Func);
                        func.body.extend(wasm::binary::CALL_INDIRECT);
                        // TODO: actually get type
                        // but rn all funcs should be [I32, I32] -> [I32] so it's ok
                        func.body.extend(
                            self.module
                                .ty_sec
                                .insert(wasm::FuncType::new(num_args, MEM_PTR_TY)),
                        );
                        func.body.extend(0x00);
                    }
                }
            }
            ast::Expr::If(if_expr) => {
                self.gen_expr(func, if_expr.condition);
                func.gen_unbox(wasm::BoxType::Bool);

                func.body.extend(wasm::binary::IF);
                // Always return a boxed ptr, even if it's nil
                func.body.extend(MEM_PTR_TY);
                self.gen_expr(func, ast::Expr::Block(if_expr.then_block));

                // Generate else block
                if let Some(else_block) = if_expr.else_block {
                    func.body.extend(wasm::binary::ELSE);

                    match else_block {
                        ast::ElseBlock::ElseIf(if_expr) => {
                            self.gen_expr(func, ast::Expr::If(if_expr))
                        }
                        ast::ElseBlock::Else(block) => self.gen_expr(func, ast::Expr::Block(block)),
                    }
                }

                func.body.extend(wasm::binary::END);
            }
            ast::Expr::Binary(binary_expr) => {
                let (op_ty, ret_ty, instrs) = {
                    use wasm::binary::{
                        ADD_F64, AND_I32, DIV_F64, EQ_F64, MUL_F64, NE_F64, OR_I32, SUB_F64,
                    };
                    use wasm::BoxType::{Bool, Num};

                    match binary_expr.op {
                        ast::BinaryOp::Or => (Bool, Bool, OR_I32),
                        ast::BinaryOp::And => (Bool, Bool, AND_I32),
                        ast::BinaryOp::NotEq => (Num, Bool, NE_F64),
                        ast::BinaryOp::Eq => (Num, Bool, EQ_F64),
                        ast::BinaryOp::Greater => todo!(),
                        ast::BinaryOp::GreaterEq => todo!(),
                        ast::BinaryOp::Less => todo!(),
                        ast::BinaryOp::LessEq => todo!(),
                        ast::BinaryOp::Subtract => (Num, Num, SUB_F64),
                        ast::BinaryOp::Add => (Num, Num, ADD_F64),
                        ast::BinaryOp::Divide => (Num, Num, DIV_F64),
                        ast::BinaryOp::Multiply => (Num, Num, MUL_F64),
                    }
                };

                self.gen_expr(func, binary_expr.rhs);
                let rhs_idx = func.gen_local_set(MEM_PTR_TY, None);

                self.gen_expr(func, binary_expr.lhs);

                func.unwrap_box(op_ty, self.mem_store.alloc(ret_ty), |func| {
                    func.gen_local_get(rhs_idx);
                    func.gen_unbox(op_ty);

                    func.body.extend(instrs)
                });
            }
            ast::Expr::Unary(unary_expr) => {
                self.gen_expr(func, unary_expr.rhs);
                match unary_expr.op {
                    ast::UnaryOp::Not => func.unwrap_box(
                        wasm::BoxType::Bool,
                        self.mem_store.alloc(wasm::BoxType::Bool),
                        |func|
                            // Use XOR 0x1 as NOT
                            // 0x0 xor 0x1 = 0x1
                            // 0x1 xor 0x1 = 0x0
                            func.body.extend([wasm::binary::CONST_I32, 0x1, wasm::binary::XOR_I32]),
                    ),
                    ast::UnaryOp::Negate => func.unwrap_box(
                        wasm::BoxType::Num,
                        self.mem_store.alloc(wasm::BoxType::Num),
                        |func| func.body.extend(wasm::binary::NEG_F64),
                    ),
                }
            }
            ast::Expr::Literal(literal) => match literal {
                ast::Literal::Bool(b) => func.gen_box(
                    self.mem_store.alloc(wasm::BoxType::Bool),
                    [|func: &mut wasm::Func| {
                        func.body.extend(wasm::binary::CONST_I32);
                        func.body.extend(b);
                    }],
                ),
                ast::Literal::Number(n) => func.gen_box(
                    self.mem_store.alloc(wasm::BoxType::Num),
                    [|func: &mut wasm::Func| {
                        func.body.extend(wasm::binary::CONST_F64);
                        func.body.extend(n);
                    }],
                ),
                ast::Literal::Str(s) => {
                    // Encode as a WasmVec of UTF-8 chars
                    let mut buf = WasmVec::new();
                    buf.extend(s.into_bytes());
                    let buf = buf.into_bytes();

                    func.gen_box(
                        self.mem_store
                            .alloc_n(wasm::BoxType::String, buf.len() as u32),
                        buf.into_iter().map(|byte| {
                            // Make sure it gets encoded w/ LEB128 and not as an opcode
                            let byte = byte as i32;
                            move |func: &mut wasm::Func| {
                                func.body.extend(wasm::binary::CONST_I32);
                                func.body.extend(byte);
                            }
                        }),
                    );
                }
                ast::Literal::Nil => func.gen_box(
                    self.mem_store.alloc(wasm::BoxType::Nil),
                    [|func: &mut wasm::Func| func.body.extend([wasm::binary::CONST_I32, 0b0])],
                ),
            },
            ast::Expr::Identifier(identifier) => func.gen_stack_get(
                &identifier
                    .location
                    .expect(&format!("location resolved for ident, {}", identifier.name)),
            ),
        }
    }
}

struct MemStore {
    mem_idx: wasm::MemIdx,
    next_idx: u32,
}

impl MemStore {
    pub fn new(mem_idx: wasm::MemIdx) -> Self {
        MemStore {
            mem_idx,
            next_idx: 0,
        }
    }

    /// # Parameters:
    /// - `box_ty`: The type of the thing being allocated.
    pub fn alloc(&mut self, box_ty: wasm::BoxType) -> MemPtr {
        self.alloc_n(box_ty, 1)
    }

    /// # Parameters:
    /// - `box_ty`: The type of the thing being allocated.
    /// - `n`: How many of the type are being allocated.
    pub fn alloc_n(&mut self, box_ty: wasm::BoxType, n: u32) -> MemPtr {
        self.alloc_raw(box_ty, n, true)
    }

    pub fn alloc_raw(&mut self, box_ty: wasm::BoxType, n: u32, includes_tag_byte: bool) -> MemPtr {
        let ptr = MemPtr {
            address: self.next_idx as i32,
            box_ty,
            n,
            includes_tag_byte,
        };
        self.next_idx += ptr.size();

        ptr
    }
}

/// The runtime WASM type of a pointer to memory.
const MEM_PTR_TY: wasm::ValType = wasm::ValType::I32;

#[derive(Clone, Copy, Debug)]
pub struct MemPtr {
    address: i32,
    box_ty: wasm::BoxType,
    n: u32,
    includes_tag_byte: bool,
}

impl MemPtr {
    /// # Panics
    /// - `n` must be less than or equal to `self.n`.
    fn offset(&self, n: u32) -> i32 {
        assert!(n <= self.n);

        let offset = n as i32;
        self.address + offset
    }

    fn size(&self) -> u32 {
        self.box_ty.size() * self.n + if self.includes_tag_byte { 1 } else { 0 }
    }
}

impl IntoBytes for MemPtr {
    fn into_bytes(self) -> Vec<u8> {
        self.address.into_bytes()
    }
}
