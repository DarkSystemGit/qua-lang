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
                    params: [wasm::ValType::I32].into_iter().collect(),
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

                let idx = self.cur_func.insert_local(wasm::ValType::I32);
                self.cur_func.body.extend(wasm::binary::LOCAL_SET);
                self.cur_func.body.extend(idx);
            }
            ast::BindingMetadata::Func {
                arguments,
                upvalues,
            } => {
                // Everything is boxed (ie an I32)
                let param_tys = arguments.iter().map(|_| wasm::ValType::I32).collect();
                let result_ty = wasm::ValType::I32;
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
            ast::Expr::Unary(unary_expr) => todo!(),
            ast::Expr::Literal(literal) => match literal {
                ast::Literal::Bool(b) => self.gen_box(MemBox::I32, |state| {
                    state.cur_func.body.extend(wasm::binary::CONST_I32);
                    state.cur_func.body.extend(b);
                }),
                ast::Literal::Number(n) => self.gen_box(MemBox::F64, |state| {
                    state.cur_func.body.extend(wasm::binary::CONST_F64);
                    state.cur_func.body.extend(n);
                }),
                ast::Literal::Str(s) => {
                    // Encode as a WasmVec of UTF-8 chars
                    let len: u32 = s.bytes().len().try_into().unwrap();
                    let len: Vec<u8> = len.into_bytes();
                    let buf = {
                        let mut buf = len;
                        buf.extend(s.into_bytes());
                        buf
                    };
                    let len = buf.len();

                    for byte in buf {
                        let byte = byte as i32;
                        self.gen_box(MemBox::I32_8U, |state| {
                            state.cur_func.body.extend(wasm::binary::CONST_I32);
                            dbg!(byte, format!("0x {:02X?}", byte.into_bytes()));
                            state.cur_func.body.extend(byte);
                        });
                    }

                    // Generate enough DROPs to remain with only the first
                    // address on the stack
                    for _ in 0..len - 1 {
                        self.cur_func.body.extend(wasm::binary::DROP);
                    }
                }
                ast::Literal::Nil => todo!(),
            },
            ast::Expr::Identifier(identifier) => todo!(),
        }
    }

    /// Boxes the top item on the stack and returns a pointer to it.
    ///
    /// `[T] -> [I32]`
    fn gen_box(&mut self, mem_box: MemBox, gen_value: impl Fn(&mut WasmGenState)) {
        let ptr = self.mem_store.alloc(mem_box.size());

        // Write the memory location
        self.cur_func.body.extend(wasm::binary::CONST_I32);
        self.cur_func.body.extend(ptr);

        // Generate the value to store
        gen_value(self);

        // Store that value in memory
        self.cur_func.body.extend(mem_box.instr_store());
        self.cur_func.body.extend([
            0x00u8, // Align 2^0=1
            0x00,   // Offset 0
        ]);

        // Return a pointer to the memory location
        self.cur_func.body.extend(wasm::binary::CONST_I32);
        self.cur_func.body.extend([ptr]);
    }

    /// Unboxes the pointer on top of the stack and returns the value.
    ///
    /// `[I32] -> [T]`
    fn gen_unbox(&mut self, mem_box: MemBox) {
        self.cur_func.body.extend(mem_box.instr_load());
        self.cur_func.body.extend([
            0x00u8, // Align 2^0=1
            0x00,   // Offset 0
        ]);
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
    /// - `size`: The size in bytes.
    pub fn alloc(&mut self, size: u32) -> MemPtr {
        let idx = MemPtr(self.next_idx as i32);
        self.next_idx += size;
        idx
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MemPtr(i32);
impl IntoBytes for MemPtr {
    fn into_bytes(self) -> Vec<u8> {
        self.0.into_bytes()
    }
}

/// The type of value to store in memory.
#[derive(Clone, Copy, Debug)]
enum MemBox {
    I32,
    I64,
    F64,

    I32_8U,
    I32_16U,
}

impl MemBox {
    pub const fn size(self) -> u32 {
        match self {
            MemBox::I32 => 32 / 8,
            MemBox::I64 => 64 / 8,
            MemBox::F64 => 64 / 8,
            MemBox::I32_8U => 8 / 8,
            MemBox::I32_16U => 16 / 8,
        }
    }

    pub const fn instr_store(self) -> impl IntoBytes {
        match self {
            MemBox::I32 => wasm::binary::MEM_I32_STORE,
            MemBox::I64 => wasm::binary::MEM_I64_STORE,
            MemBox::F64 => wasm::binary::MEM_F64_STORE,
            MemBox::I32_8U => wasm::binary::MEM_I32_STORE_8,
            MemBox::I32_16U => wasm::binary::MEM_I32_STORE_16,
        }
    }

    pub const fn instr_load(self) -> impl IntoBytes {
        match self {
            MemBox::I32 => wasm::binary::MEM_I32_LOAD,
            MemBox::I64 => wasm::binary::MEM_I64_LOAD,
            MemBox::F64 => wasm::binary::MEM_F64_LOAD,
            MemBox::I32_8U => wasm::binary::MEM_I32_LOAD_8U,
            MemBox::I32_16U => wasm::binary::MEM_I32_LOAD_16U,
        }
    }
}
