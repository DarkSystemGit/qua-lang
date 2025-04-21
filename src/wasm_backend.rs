use wasm::binary::{IntoBytes, WasmVec};

use crate::ast;

mod wasm;

// Whether or not to generate code to check types of boxes at runtime.
const CHECK_TYPES: bool = true;

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

        let global_mem_alloc_ptr = module.globals_sec.insert(
            wasm::Global {
                ty: MEM_PTR_TY,
                mutable: true,
                init: {
                    let mut expr = wasm::binary::Expr::new();
                    expr.extend([wasm::binary::CONST_I32, 0x00]);
                    expr
                },
            },
            Some((
                &mut module.name_sec,
                wasm::Name("<mem_alloc_ptr>".to_string()),
            )),
        );

        let mut main_func = {
            let ty = wasm::FuncType {
                params: WasmVec::new(),
                results: WasmVec::new(),
            };
            let ty = module.ty_sec.insert(ty);
            wasm::Func::new_base(ty, [])
        };

        let mut state = WasmGenState {
            module,
            mem_store: MemStore::new(mem_idx, global_mem_alloc_ptr),
        };

        // Add imports vars to main func.
        // Assumes that import indexes are in order (`.enumerate()`), and that
        // new imports will not be added after/during this loop (`.clone()`).
        for (i, import) in state.module.funcs.imports().clone().iter().enumerate() {
            // TODO: actually track # of args + result
            let ty = wasm::FuncType::new(1, MEM_PTR_TY);
            let ty = state.module.ty_sec.insert(ty);
            let mut func = wasm::Func::new(ty, Some(import.dbg_name()), [None], &[]);
            // Assumes first arg is at index 1
            func.gen_stack_get(&ast::IdentLocation::Stack(ast::StackIndex(1)));
            func.body.extend(wasm::binary::CALL);
            func.body.extend(i as u32);
            state.gen_boxed_nil(&mut func);

            state.gen_func_def(&mut main_func, func, [], Some(import.dbg_name()));
            main_func.gen_local_set(
                MEM_PTR_TY,
                // Assumes that the imports are in order in the ast stack
                Some(ast::IdentLocation::Stack(ast::StackIndex(i))),
                Some(import.dbg_name()),
            );
        }

        state.gen_program(&mut main_func, program);
        state.finish(main_func)
    }

    fn finish(mut self, func: wasm::Func) -> Vec<u8> {
        // Set up main function
        let main_idx = self.module.funcs.insert(
            func,
            &mut self.module.name_sec,
            Some(wasm::Name("<main>".to_string())),
        );

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

                // *all* expressions must generate some return value, even if
                // it's nil. Therefore, they must all be dropped.
                func.body.extend(wasm::binary::DROP);
            }
        }
    }

    fn gen_binding(&mut self, func: &mut wasm::Func, binding: ast::Binding) {
        let dbg_name = Some(wasm::Name(binding.ident.name.clone()));
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
                let mut new_func = wasm::Func::new(
                    ty,
                    dbg_name.clone(),
                    arguments.into_iter().map(Some),
                    upvalues.as_slice(),
                );

                self.gen_expr(&mut new_func, binding.value);

                self.gen_func_def(func, new_func, upvalues, dbg_name.clone());
            }
        }

        func.gen_local_set(
            MEM_PTR_TY,
            Some(
                binding.ident.location.unwrap_or_else(|| {
                    panic!("location resolved for ident, {}", binding.ident.name)
                }),
            ),
            dbg_name,
        );
    }

    fn gen_func_def<I: IntoIterator<Item = ast::Upvalue>>(
        &mut self,
        func: &mut wasm::Func,
        new_func: wasm::Func,
        upvalues: I,
        dbg_name: Option<wasm::Name>,
    ) where
        I::IntoIter: ExactSizeIterator,
    {
        let upvalues = upvalues.into_iter();
        let num_upvalues = upvalues.len() as u32;

        let idx = self
            .module
            .funcs
            .insert(new_func, &mut self.module.name_sec, dbg_name);
        let idx = self.module.funcs.raw_func_sec_idx(idx);
        let ptr = self
            .mem_store
            .alloc_n(func, wasm::BoxType::Func, 1 + num_upvalues);

        let upvalues = upvalues
            .map(|upvalue| {
                func.gen_stack_get(&upvalue.target);
                func.gen_local_set(MEM_PTR_TY, None, Some(wasm::Name(upvalue.dbg_name)))
            })
            .collect::<Vec<_>>();
        let upvalues = &upvalues;

        func.gen_box(
            ptr,
            (0..=num_upvalues)
                .map(|i| {
                    move |func: &mut wasm::Func| match i {
                        0 => {
                            func.body.extend(wasm::binary::CONST_I32);
                            func.body.extend(idx);
                        }
                        i => {
                            let local_idx = upvalues[i as usize - 1];
                            func.gen_local_get(local_idx);
                        }
                    }
                })
                .collect::<Vec<_>>(),
        );
    }

    fn gen_expr(&mut self, func: &mut wasm::Func, expr: ast::Expr) {
        // NOTE: all expressions must return some value, even if it is nil.
        match expr {
            ast::Expr::Block(block) => {
                for stmt in block.stmts {
                    self.gen_stmt(func, stmt);
                }

                if let Some(return_expr) = block.return_expr {
                    self.gen_expr(func, *return_expr);
                } else {
                    self.gen_boxed_nil(func);
                }
            }
            ast::Expr::Call(call) => {
                // Put arguments onto the stack
                // Start with self reference
                self.gen_expr(func, *call.target);
                let target_idx = func.gen_local_tee(MEM_PTR_TY, None, None);
                // Then the real args
                // Save this before call.arguments is consumed in the for loop
                let num_args = call.arguments.len();
                for arg in call.arguments {
                    self.gen_expr(func, arg);
                }

                // Actually call the function
                func.gen_local_get(target_idx);
                func.gen_unbox(wasm::BoxType::Func);
                func.body.extend(wasm::binary::CALL_INDIRECT);
                func.body.extend(
                    self.module
                        .ty_sec
                        .insert(wasm::FuncType::new(num_args, MEM_PTR_TY)),
                );
                func.body.extend(0x00); // Last arg to CALL_INDIRECT (the index of the table)
            }
            ast::Expr::If(if_expr) => {
                self.gen_expr(func, if_expr.condition);
                func.gen_unbox(wasm::BoxType::Bool);

                func.body.extend(wasm::binary::IF);
                // Always return a boxed ptr, even if it's nil
                func.body.extend(MEM_PTR_TY);
                self.gen_expr(func, ast::Expr::Block(if_expr.then_block));

                // Generate else block
                func.body.extend(wasm::binary::ELSE);

                if let Some(else_block) = if_expr.else_block {
                    match else_block {
                        ast::ElseBlock::ElseIf(if_expr) => {
                            self.gen_expr(func, ast::Expr::If(if_expr))
                        }
                        ast::ElseBlock::Else(block) => self.gen_expr(func, ast::Expr::Block(block)),
                    }
                } else {
                    self.gen_boxed_nil(func);
                }

                func.body.extend(wasm::binary::END);
            }
            ast::Expr::Binary(binary_expr) => {
                let (op_ty, ret_ty, instrs) = {
                    use wasm::binary::{
                        ADD_F64, AND_I32, DIV_F64, EQ_F64, GE_F64, GT_F64, LE_F64, LT_F64, MUL_F64,
                        NE_F64, OR_I32, SUB_F64,
                    };
                    use wasm::BoxType::{Bool, Num};

                    match binary_expr.op {
                        ast::BinaryOp::Or => (Bool, Bool, OR_I32),
                        ast::BinaryOp::And => (Bool, Bool, AND_I32),
                        ast::BinaryOp::NotEq => (Num, Bool, NE_F64),
                        ast::BinaryOp::Eq => (Num, Bool, EQ_F64),
                        ast::BinaryOp::Greater => (Num, Bool, GT_F64),
                        ast::BinaryOp::GreaterEq => (Num, Bool, GE_F64),
                        ast::BinaryOp::Less => (Num, Bool, LT_F64),
                        ast::BinaryOp::LessEq => (Num, Bool, LE_F64),
                        ast::BinaryOp::Subtract => (Num, Num, SUB_F64),
                        ast::BinaryOp::Add => (Num, Num, ADD_F64),
                        ast::BinaryOp::Divide => (Num, Num, DIV_F64),
                        ast::BinaryOp::Multiply => (Num, Num, MUL_F64),
                    }
                };

                self.gen_expr(func, binary_expr.rhs);
                let rhs_idx = func.gen_local_set(MEM_PTR_TY, None, None);

                self.gen_expr(func, binary_expr.lhs);

                let rebox_ptr = self.mem_store.alloc(func, ret_ty);
                func.unwrap_box(op_ty, rebox_ptr, |func| {
                    func.gen_local_get(rhs_idx);
                    func.gen_unbox(op_ty);

                    func.body.extend(instrs)
                });
            }
            ast::Expr::Unary(unary_expr) => {
                self.gen_expr(func, unary_expr.rhs);
                match unary_expr.op {
                    ast::UnaryOp::Not => {
                        let rebox_ptr = self.mem_store.alloc(func, wasm::BoxType::Bool);
                        func.unwrap_box(wasm::BoxType::Bool, rebox_ptr, |func| {
                            // Use XOR 0x1 as NOT
                            // 0x0 xor 0x1 = 0x1
                            // 0x1 xor 0x1 = 0x0
                            func.body
                                .extend([wasm::binary::CONST_I32, 0x1, wasm::binary::XOR_I32])
                        })
                    }
                    ast::UnaryOp::Negate => {
                        let rebox_ptr = self.mem_store.alloc(func, wasm::BoxType::Num);
                        func.unwrap_box(wasm::BoxType::Num, rebox_ptr, |func| {
                            func.body.extend(wasm::binary::NEG_F64)
                        })
                    }
                }
            }
            ast::Expr::Literal(literal) => match literal {
                ast::Literal::Bool(b) => {
                    let ptr = self.mem_store.alloc(func, wasm::BoxType::Bool);
                    func.gen_box(
                        ptr,
                        [|func: &mut wasm::Func| {
                            func.body.extend(wasm::binary::CONST_I32);
                            func.body.extend(b);
                        }],
                    )
                }
                ast::Literal::Number(n) => {
                    let ptr = self.mem_store.alloc(func, wasm::BoxType::Num);
                    func.gen_box(
                        ptr,
                        [|func: &mut wasm::Func| {
                            func.body.extend(wasm::binary::CONST_F64);
                            func.body.extend(n);
                        }],
                    )
                }
                ast::Literal::Str(s) => {
                    // Encode as a WasmVec of UTF-8 chars
                    let mut buf = WasmVec::new();
                    buf.extend(s.into_bytes());
                    let buf = buf.into_bytes();

                    let ptr = self
                        .mem_store
                        .alloc_n(func, wasm::BoxType::String, buf.len() as u32);
                    func.gen_box(
                        ptr,
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
                ast::Literal::Nil => self.gen_boxed_nil(func),
            },
            ast::Expr::Identifier(identifier) => func.gen_stack_get(
                &identifier
                    .location
                    .unwrap_or_else(|| panic!("location resolved for ident, {}", identifier.name)),
            ),
        }
    }

    fn gen_boxed_nil(&mut self, func: &mut wasm::Func) {
        let ptr = self.mem_store.alloc(func, wasm::BoxType::Nil);
        func.gen_box(
            ptr,
            [|func: &mut wasm::Func| func.body.extend([wasm::binary::CONST_I32, 0b0])],
        )
    }
}

struct MemStore {
    mem_idx: wasm::MemIdx,
    global_mem_alloc_ptr: wasm::GlobalIdx,
}

impl MemStore {
    pub fn new(mem_idx: wasm::MemIdx, global_mem_alloc_ptr: wasm::GlobalIdx) -> Self {
        MemStore {
            mem_idx,
            global_mem_alloc_ptr,
        }
    }

    /// # Parameters:
    /// - `box_ty`: The type of the thing being allocated.
    pub fn alloc(&mut self, func: &mut wasm::Func, box_ty: wasm::BoxType) -> MemPtr {
        self.alloc_n(func, box_ty, 1)
    }

    /// # Parameters:
    /// - `box_ty`: The type of the thing being allocated.
    /// - `n`: How many of the type are being allocated.
    pub fn alloc_n(&mut self, func: &mut wasm::Func, box_ty: wasm::BoxType, n: u32) -> MemPtr {
        self.alloc_raw(func, box_ty, n, true)
    }

    pub fn alloc_raw(
        &mut self,
        func: &mut wasm::Func,
        box_ty: wasm::BoxType,
        n: u32,
        includes_tag_byte: bool,
    ) -> MemPtr {
        // Get mem alloc ptr
        func.body.extend(wasm::binary::GLOBAL_GET);
        func.body.extend(self.global_mem_alloc_ptr);

        // Store address to local
        let local_idx = func.gen_local_tee(MEM_PTR_TY, None, None);
        let ptr = MemPtr {
            local_idx,
            box_ty,
            n,
            includes_tag_byte,
        };

        // Update the global mem alloc ptr
        // Add the size to it
        func.body.extend(wasm::binary::CONST_I32);
        func.body.extend(ptr.size());
        func.body.extend(wasm::binary::ADD_I32);
        // Update it
        func.body.extend(wasm::binary::GLOBAL_SET);
        func.body.extend(self.global_mem_alloc_ptr);

        ptr
    }
}

/// The runtime WASM type of a pointer to memory.
const MEM_PTR_TY: wasm::ValType = wasm::ValType::I32;

#[derive(Clone, Copy, Debug)]
pub struct MemPtr {
    local_idx: wasm::LocalIdx,
    box_ty: wasm::BoxType,
    n: u32,
    includes_tag_byte: bool,
}

impl MemPtr {
    // Puts the pointer onto the stack.
    //
    // # Stack
    //
    // `[] -> [MEM_PTR_TY]`
    fn gen_load(&self, func: &mut wasm::Func) {
        func.gen_local_get(self.local_idx);
    }

    /// Loads the ptr, offsets it, and puts it on the stack.
    ///
    /// # Stack
    ///
    /// `[] -> [MEM_PTR_TY]`
    ///
    /// # Panics
    /// - `n` must be less than or equal to `self.n`.
    fn gen_offset(&self, func: &mut wasm::Func, n: u32) {
        assert!(n <= self.n);

        func.gen_local_get(self.local_idx);

        let offset = self.includes_tag_byte as u32 + self.box_ty.size() * n;
        if offset != 0 {
            func.body.extend(wasm::binary::CONST_I32);
            func.body.extend(offset);
            func.body.extend(wasm::binary::ADD_I32);
        }
    }

    const fn func_self_ref(num_upvalues: u32) -> Self {
        MemPtr {
            local_idx: wasm::LocalIdx::FUNC_SELF_REF,
            box_ty: wasm::BoxType::Func,
            n: 1 + num_upvalues,
            includes_tag_byte: true,
        }
    }

    fn size(&self) -> u32 {
        self.box_ty.size() * self.n + if self.includes_tag_byte { 1 } else { 0 }
    }
}
