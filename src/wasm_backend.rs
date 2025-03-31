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
                self.gen_local_set(MEM_PTR_TY);
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
                    ast::UnaryOp::Not => self.unwrap_box(
                        BoxType::Bool,
                        BoxType::Bool,
                        |state| 
                            // Use XOR 0x1 as NOT
                            // 0x0 xor 0x1 = 0x1
                            // 0x1 xor 0x1 = 0x0
                            state.cur_func.body.extend([wasm::binary::CONST_I32, 0x1, wasm::binary::XOR_I32])
                    ),
                    ast::UnaryOp::Negate => self.unwrap_box(
                        BoxType::Num,
                        BoxType::Num,
                        |state| state.cur_func.body.extend(wasm::binary::NEG_F64)
                    ),
                }
            }
            ast::Expr::Literal(literal) => match literal {
                ast::Literal::Bool(b) => self.gen_box(
                    BoxType::Bool,
                    [|state: &mut Self| {
                        state.cur_func.body.extend(wasm::binary::CONST_I32);
                        state.cur_func.body.extend(b);
                    }],
                ),
                ast::Literal::Number(n) => self.gen_box(
                    BoxType::Num,
                    [|state: &mut Self| {
                        state.cur_func.body.extend(wasm::binary::CONST_F64);
                        state.cur_func.body.extend(n);
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

                    self.gen_box(
                        BoxType::Byte,
                        buf.into_iter().map(|byte| {
                            // Make sure it gets encoded w/ LEB128 and not as an opcode
                            let byte = byte as i32;
                            move |state: &mut Self| {
                                state.cur_func.body.extend(wasm::binary::CONST_I32);
                                state.cur_func.body.extend(byte);
                            }
                        }),
                    );
                }
                ast::Literal::Nil => todo!(),
            },
            ast::Expr::Identifier(identifier) => todo!(),
        }
    }

    /// Places the value on the top of the stack into a new local.
    ///
    /// `[T] -> []`
    fn gen_local_set(&mut self, ty: wasm::ValType) -> wasm::LocalIdx {
        let idx = self.cur_func.insert_local(ty);
        self.cur_func.body.extend(wasm::binary::LOCAL_SET);
        self.cur_func.body.extend(idx);
        idx
    }

    /// Reads from a local and places it on the stack.
    ///
    /// `[] -> [T]`
    fn gen_local_get(&mut self, idx: wasm::LocalIdx) {
        self.cur_func.body.extend(wasm::binary::LOCAL_GET);
        self.cur_func.body.extend(idx);
    }

    /// Boxes the top item on the stack and returns a pointer to it.
    ///
    /// `[T] -> [I32]`
    fn gen_box<T: Fn(&mut WasmGenState), I: IntoIterator<Item = T>>(
        &mut self,
        box_ty: BoxType,
        gen_values: I,
        // gen_values: &[impl Fn(&mut WasmGenState)],
    ) where
        I::IntoIter: ExactSizeIterator,
    {
        let gen_values = gen_values.into_iter();
        let len: u32 = gen_values.len().try_into().unwrap();

        let ptr = self.mem_store.alloc(box_ty.size() * len);
        let mut offset = 0;

        for gen_value in gen_values {
            // Write the memory location
            self.cur_func.body.extend(wasm::binary::CONST_I32);
            self.cur_func.body.extend(ptr.offset(offset));

            // Generate the value to store
            gen_value(self);

            // Store that value in memory
            self.cur_func.body.extend(box_ty.instr_store());
            self.cur_func.body.extend([
                0x00u8, // Align 2^0=1
                0x00,   // Offset 0
            ]);

            // Increase offset
            offset += box_ty.size() as i32;
        }

        // Return a pointer to the memory location
        self.cur_func.body.extend(wasm::binary::CONST_I32);
        self.cur_func.body.extend([ptr]);
    }

    /// Unboxes the pointer on top of the stack and returns the value.
    ///
    /// `[I32] -> [T]`
    fn gen_unbox(&mut self, box_ty: BoxType) {
        self.cur_func.body.extend(box_ty.instr_load());
        self.cur_func.body.extend([
            0x00u8, // Align 2^0=1
            0x00,   // Offset 0
        ]);
    }

    /// Unboxes the pointer on top of the stack, runs `func()` to work with it,
    /// and then reboxes the top of the stack.
    ///
    /// `[I32] -> [I32]`
    ///
    /// # Parameters
    /// - `mem_unbox`: What to unbox into before `func()` is run.
    /// - `mem_rebox`: What to rebox into after `func()` is run.
    /// - `func`: The function to run w/ the unboxed value. Must be `[Unbox] -> [Rebox]`.
    fn unwrap_box(
        &mut self,
        unbox_ty: BoxType,
        rebox_ty: BoxType,
        func: impl Fn(&mut WasmGenState),
    ) {
        self.gen_unbox(unbox_ty);

        func(self);
        let res_idx = self.gen_local_set(rebox_ty.into());

        self.gen_box(
            rebox_ty,
            [|state: &mut Self| {
                state.gen_local_get(res_idx);
            }],
        );
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

/// The type of value to store in memory.
#[derive(Clone, Copy, Debug)]
enum BoxType {
    Num,
    Bool,
    Byte,
}

impl BoxType {
    pub const fn size(self) -> u32 {
        match self {
            BoxType::Num => 64 / 8,
            BoxType::Bool => 8 / 8,
            BoxType::Byte => 8 / 8,
        }
    }

    pub const fn instr_store(self) -> impl IntoBytes {
        match self {
            BoxType::Num => wasm::binary::MEM_F64_STORE,
            BoxType::Bool => wasm::binary::MEM_I32_STORE_8,
            BoxType::Byte => wasm::binary::MEM_I32_STORE_8,
        }
    }

    pub const fn instr_load(self) -> impl IntoBytes {
        match self {
            BoxType::Num => wasm::binary::MEM_F64_LOAD,
            BoxType::Bool => wasm::binary::MEM_I32_LOAD_8U,
            BoxType::Byte => wasm::binary::MEM_I32_LOAD_8U,
        }
    }
}

impl From<BoxType> for wasm::ValType {
    fn from(value: BoxType) -> Self {
        use wasm::ValType;
        match value {
            BoxType::Num => ValType::F64,
            BoxType::Bool => ValType::I32,
            BoxType::Byte => ValType::I32,
        }
    }
}
