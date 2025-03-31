use wasm::binary::{IntoBytes, WasmVec};

use crate::ast;

mod wasm;

pub fn gen_wasm(program: ast::Program) -> Vec<u8> {
    let mut state = WasmGenState::new();
    state.gen_program(program);
    state.finish()
}

struct WasmGenState {
    module: wasm::Module,
    cur_func: wasm::Func,
    mem_store: MemStore,
}
impl WasmGenState {
    fn new() -> Self {
        let mut module = wasm::Module::default();

        let import_host_print = wasm::FuncImport {
            module: wasm::Name("host".to_string()),
            name: wasm::Name("print".to_string()),
            ty: {
                let ty = wasm::FuncType {
                    // TODO: funcs
                    params: [MEM_PTR_TY].into_iter().collect(),
                    results: WasmVec::new(),
                };
                module.ty_sec.insert(ty)
            },
        };
        module.funcs.insert_import(import_host_print);

        let mem_idx = module.mem_sec.insert(wasm::MemType {
            limits: wasm::Limits { min: 64, max: None },
        });

        let main_func = {
            let ty = wasm::FuncType {
                params: WasmVec::new(),
                results: WasmVec::new(),
            };
            let ty = module.ty_sec.insert(ty);
            wasm::Func::new(ty)
        };

        WasmGenState {
            module,
            cur_func: main_func,
            mem_store: MemStore::new(mem_idx),
        }
    }

    fn finish(mut self) -> Vec<u8> {
        // Set up main function
        let main_idx = self.module.funcs.insert(self.cur_func);

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

        self.module.into_bytes()
    }

    fn gen_program(&mut self, program: ast::Program) {
        for stmt in program {
            self.gen_stmt(stmt);
        }
    }

    fn gen_stmt(&mut self, stmt: ast::Stmt) {
        match stmt {
            ast::Stmt::Let(binding) => self.gen_binding(binding),
            ast::Stmt::Expr(expr) => self.gen_expr(expr),
        }
    }

    fn gen_binding(&mut self, binding: ast::Binding) {
        match binding.metadata {
            ast::BindingMetadata::Var => {
                self.gen_expr(binding.value);
                self.cur_func.gen_local_set(MEM_PTR_TY);
            }
            ast::BindingMetadata::Func {
                arguments,
                upvalues,
            } => {
                let param_tys = arguments.iter().map(|_| MEM_PTR_TY).collect();
                let result_ty = MEM_PTR_TY;
                let ty = wasm::FuncType {
                    params: param_tys,
                    // Functions can only have 1 return type as of now
                    results: [result_ty].into_iter().collect(),
                };

                let ty = self.module.ty_sec.insert(ty);
                let func = wasm::Func::new(ty);

                // TODO: actual generate body

                self.module.funcs.insert(func);
            }
        }
    }

    fn gen_expr(&mut self, expr: ast::Expr) {
        match expr {
            ast::Expr::Block(block) => {
                for stmt in block.stmts {
                    self.gen_stmt(stmt);
                }

                if let Some(return_expr) = block.return_expr {
                    self.gen_expr(*return_expr);
                }
            }
            ast::Expr::Call(call) => {
                // Put arguments onto the stack
                for arg in call.arguments {
                    self.gen_expr(arg);
                }

                // TODO: funcs
                self.cur_func.body.extend(wasm::binary::CALL);
                self.cur_func.body.extend(0u32);
            }
            ast::Expr::If(if_expr) => todo!(),
            ast::Expr::Binary(binary_expr) => todo!(),
            ast::Expr::Unary(unary_expr) => {
                self.gen_expr(unary_expr.rhs);
                match unary_expr.op {
                    ast::UnaryOp::Not => self.cur_func.unwrap_box(
                        wasm::BoxType::Bool,
                        self.mem_store.alloc(wasm::BoxType::Bool),
                        |func: &mut wasm::Func| 
                            // Use XOR 0x1 as NOT
                            // 0x0 xor 0x1 = 0x1
                            // 0x1 xor 0x1 = 0x0
                            func.body.extend([wasm::binary::CONST_I32, 0x1, wasm::binary::XOR_I32])
                    ),
                    ast::UnaryOp::Negate => self.cur_func.unwrap_box(
                        wasm::BoxType::Num,
                        self.mem_store.alloc(wasm::BoxType::Num),
                        |func: &mut wasm::Func| func.body.extend(wasm::binary::NEG_F64)
                    ),
                }
            }
            ast::Expr::Literal(literal) => match literal {
                ast::Literal::Bool(b) => self.cur_func.gen_box(
                    self.mem_store.alloc(wasm::BoxType::Bool),
                    [|func: &mut wasm::Func| {
                        func.body.extend(wasm::binary::CONST_I32);
                        func.body.extend(b);
                    }],
                ),
                ast::Literal::Number(n) => self.cur_func.gen_box(
                    self.mem_store.alloc(wasm::BoxType::Num),
                    [|func: &mut wasm::Func| {
                        func.body.extend(wasm::binary::CONST_F64);
                        func.body.extend(n);
                    }],
                ),
                ast::Literal::Str(s) => {
                    // Encode as a WasmVec of UTF-8 chars
                    let len: u32 = s.bytes().len().try_into().unwrap();
                    let len: Vec<u8> = len.into_bytes();
                    let buf = {
                        let mut buf = len;
                        buf.extend(s.into_bytes());
                        buf
                    };

                    self.cur_func.gen_box(
                        self.mem_store.alloc(wasm::BoxType::Byte),
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
                ast::Literal::Nil => todo!(),
            },
            ast::Expr::Identifier(identifier) => todo!(),
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
    pub fn alloc(&mut self, box_ty: wasm::BoxType) -> (MemPtr, wasm::BoxType) {
        self.alloc_n(box_ty, 1)
    }

    /// # Parameters:
    /// - `box_ty`: The type of the thing being allocated.
    /// - `n`: How many of the type are being allocated.
    pub fn alloc_n(&mut self, box_ty: wasm::BoxType, n: u32) -> (MemPtr, wasm::BoxType) {
        let idx = MemPtr(self.next_idx as i32);
        self.next_idx += box_ty.size() * n;
        (idx, box_ty)
    }
}

/// The runtime WASM type of a pointer to memory.
const MEM_PTR_TY: wasm::ValType = wasm::ValType::I32;

#[derive(Clone, Copy, Debug)]
pub struct MemPtr(i32);
impl MemPtr {
    fn offset(mut self, amount: i32) -> MemPtr {
        self.0 += amount;
        self
    }
}
impl IntoBytes for MemPtr {
    fn into_bytes(self) -> Vec<u8> {
        self.0.into_bytes()
    }
}

