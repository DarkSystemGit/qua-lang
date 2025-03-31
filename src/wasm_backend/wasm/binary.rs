use std::marker::PhantomData;

pub const MAGIC_NUM: [u8; 4] = *b"\0asm";
pub const VERSION: [u8; 4] = [0x01, 0x00, 0x00, 0x00];

// Section types
pub const SEC_CUSTOM: u8 = 0x00;
pub const SEC_TY: u8 = 0x01;
pub const SEC_IMPORT: u8 = 0x02;
pub const SEC_FUNC: u8 = 0x03;
pub const SEC_TABLE: u8 = 0x04;
pub const SEC_MEM: u8 = 0x05;
pub const SEC_GLOBAL: u8 = 0x06;
pub const SEC_EXPORT: u8 = 0x07;
pub const SEC_START: u8 = 0x08;
pub const SEC_ELEM: u8 = 0x09;
pub const SEC_CODE: u8 = 0x0A;
pub const SEC_DATA: u8 = 0x0B;
pub const SEC_DATA_COUNT: u8 = 0x0C;

// Number types
pub const TY_I32: u8 = 0x7F;
pub const TY_I64: u8 = 0x7E;
pub const TY_F32: u8 = 0x7D;
pub const TY_F64: u8 = 0x7C;

// Vector type
pub const TY_VEC: u8 = 0x7B;

// Reference types
pub const TY_FUNC_REF: u8 = 0x70;
pub const TY_EXTERN_REF: u8 = 0x6F;

// Function type
pub const TY_FUNC: u8 = 0x60;

// Control instructions
pub const END: u8 = 0x0B;
pub const CALL: u8 = 0x10;
pub const DROP: u8 = 0x1A;

// Variable instructions
pub const LOCAL_GET: u8 = 0x20;
pub const LOCAL_SET: u8 = 0x21;
pub const LOCAL_TEE: u8 = 0x22;

// Memory instructions
pub const MEM_I32_LOAD: u8 = 0x28;
pub const MEM_I32_STORE: u8 = 0x36;
pub const MEM_I64_LOAD: u8 = 0x29;
pub const MEM_I64_STORE: u8 = 0x37;
pub const MEM_F64_LOAD: u8 = 0x2B;
pub const MEM_F64_STORE: u8 = 0x39;
pub const MEM_I32_LOAD_8U: u8 = 0x2D;
pub const MEM_I32_STORE_8: u8 = 0x3A;
pub const MEM_I32_LOAD_16U: u8 = 0x2F;
pub const MEM_I32_STORE_16: u8 = 0x3B;

// Numeric
pub const CONST_I32: u8 = 0x41;
pub const CONST_I64: u8 = 0x42;
pub const CONST_F32: u8 = 0x43;
pub const CONST_F64: u8 = 0x44;

pub const SUB_I32: u8 = 0x6B;

/// Turn a section into bytecode with a proper header.
///
/// See <https://webassembly.github.io/spec/core/binary/modules.html#sections>.
pub fn sec_bytes(sec_id: u8, contents: impl IntoBytes) -> Vec<u8> {
    let contents = contents.into_bytes();

    let mut buf = vec![sec_id];

    let size = contents.len() as u32;
    buf.extend(size.into_bytes());

    buf.extend(contents);

    buf
}

pub trait IntoBytes {
    fn into_bytes(self) -> Vec<u8>;
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WasmVec<T> {
    vec: Vec<u8>,
    size: u32,
    _phantom: PhantomData<T>,
}

impl<T> WasmVec<T> {
    pub fn new() -> Self {
        WasmVec {
            vec: Vec::new(),
            size: 0,
            _phantom: PhantomData,
        }
    }

    pub fn extend<I: IntoIterator<Item = T>>(&mut self, items: I)
    where
        T: IntoBytes,
    {
        self.vec.extend(items.into_iter().flat_map(|i| {
            self.size += 1;
            i.into_bytes()
        }))
    }

    pub fn size(&self) -> u32 {
        self.size
    }
}

impl<T> IntoBytes for WasmVec<T> {
    fn into_bytes(self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Write the header
        buf.extend(self.size.into_bytes());

        // Write the actual bytes
        buf.extend(self.vec);

        buf
    }
}

impl<T> Default for WasmVec<T> {
    fn default() -> Self {
        WasmVec::new()
    }
}

impl<T> std::ops::Deref for WasmVec<T> {
    type Target = Vec<u8>;

    fn deref(&self) -> &Self::Target {
        &self.vec
    }
}

impl<A: IntoBytes> FromIterator<A> for WasmVec<A> {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        let mut vec = WasmVec::new();
        vec.extend(iter);
        vec
    }
}

/// The raw bytes of the instructions. Shoud not include the END opcode.
#[derive(Debug)]
pub struct Expr {
    instructions: Vec<u8>,
}

impl Expr {
    pub fn new() -> Self {
        Expr {
            instructions: Vec::new(),
        }
    }

    pub fn extend(&mut self, instr: impl IntoBytes) {
        self.instructions.extend(instr.into_bytes());
    }
}

impl IntoBytes for Expr {
    fn into_bytes(self) -> Vec<u8> {
        let mut buf = self.instructions;
        buf.push(END);
        buf
    }
}

impl IntoBytes for u32 {
    fn into_bytes(self) -> Vec<u8> {
        let mut buf = Vec::new();
        leb128::write::unsigned(&mut buf, self.into()).unwrap();
        buf
    }
}

impl IntoBytes for i32 {
    fn into_bytes(self) -> Vec<u8> {
        let mut buf = Vec::new();
        leb128::write::signed(&mut buf, self.into()).unwrap();
        buf
    }
}

impl IntoBytes for u8 {
    fn into_bytes(self) -> Vec<u8> {
        vec![self]
    }
}

impl IntoBytes for f64 {
    fn into_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl IntoBytes for bool {
    fn into_bytes(self) -> Vec<u8> {
        vec![self as u8]
    }
}

impl<const N: usize, T: IntoBytes> IntoBytes for [T; N] {
    fn into_bytes(self) -> Vec<u8> {
        self.into_iter().flat_map(|e| e.into_bytes()).collect()
    }
}

impl<T: IntoBytes + Copy> IntoBytes for &[T] {
    fn into_bytes(self) -> Vec<u8> {
        self.iter().flat_map(|e| e.into_bytes()).collect()
    }
}
