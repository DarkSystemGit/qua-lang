use std::{
    fs,
    io::{self, Write},
    path::Path,
};

use wasm_backend::gen_wasm;

mod ast;
mod interperter;
mod lexer;
mod parser;
mod stream;
mod util;
mod wasm_backend;

fn main() {
    let mut args = std::env::args();

    let _ = args.next();
    match args.next() {
        Some(wasm_flag) if wasm_flag == "--wasm" => {
            // FIXME: clean this up

            let Some(path) = args.next() else {
                eprintln!("Error: must provide file to compile.");
                return;
            };

            let ast = match ast_from_file(path.clone()) {
                Ok(ast) => ast,
                Err(err) => return eprintln!("Error parsing AST: {err:#?}"),
            };
            let wasm = gen_wasm(ast);

            let path = Path::new(&path);
            let Some(std::path::Component::Normal(filename)) = path.components().last() else {
                eprintln!("Error: path should contain filename");
                return;
            };
            let filename = filename
                .to_os_string()
                .into_string()
                .expect("filename should be valid utf8");
            let out_file_name = format!("../wasm-runner/{filename}.wasm");
            let out_file_path = Path::new(&out_file_name);
            let mut out_file = match std::fs::File::create(out_file_path) {
                Ok(f) => f,
                Err(err) => return eprintln!("Error creating file {out_file_path:#?}: {err:#?}"),
            };
            match out_file.write_all(&wasm.into_bytes()) {
                Ok(()) => eprintln!("Wrote to {out_file_path:#?}"),
                Err(err) => eprintln!("Error writing to file: {err}"),
            }
        }
        Some(path) => {
            let source = fs::read_to_string(path).expect("source file is readable");
            let mut env = (parser::Env::new(), interperter::Env::new());
            run(source, &mut env);
        }
        None => {
            run_repl();
        }
    }
}

fn run_repl() {
    // start repl
    eprintln!("qua v0.1.0");
    let mut env = (parser::Env::new(), interperter::Env::new());
    eprint!("> ");
    while let Some(Ok(line)) = io::stdin().lines().next() {
        run(line, &mut env);
        eprint!("> ");
    }
    eprintln!("Goodbye! o/");
}

fn ast_from_file(path: String) -> parser::Parse<ast::Program> {
    let source = fs::read_to_string(path).expect("source file is readable");
    let mut env = (parser::Env::new(), interperter::Env::new());

    let tokens = lexer::lex(source.clone());
    parser::parse(tokens, &mut env.0)
}

fn run(source: String, env: &mut (parser::Env, interperter::Env)) {
    let tokens = lexer::lex(source.clone());
    // println!("{tokens:#?}");

    let ast = parser::parse(tokens, &mut env.0);
    // dbg!(&ast);

    match ast {
        Ok(ast) => {
            let res = interperter::interpert(ast, &mut env.1);
            if let Err(err) = res {
                let msg = display_error(err, source);
                eprintln!("{msg}");
            }
        }
        Err(err) => println!("Error while parsing AST: {err:#?}"),
    }
}

fn display_error(err: interperter::Error, source: String) -> String {
    format!(
        "Error{}: {:#?}",
        if let Some(pos) = err.pos {
            let (line, col) = pos.calculate_line_col(&source);
            format!(" at {line}:{col}")
        } else {
            "".to_string()
        },
        err.kind
    )
}

impl lexer::Pos {
    /// (line, col)
    ///
    /// e.g.
    /// abc
    /// def
    /// f -> (2, 3)
    fn calculate_line_col(&self, source: &str) -> (usize, usize) {
        let mut line = 1;
        let mut col = 0;
        for c in source.chars().take(self.0) {
            col += 1;
            if c == '\n' {
                line += 1;
                col = 0;
            }
        }
        (line, col)
    }
}
