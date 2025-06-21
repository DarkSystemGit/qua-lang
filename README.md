# qua

This is my little toy programming language. It's impure and fully immutable, so
ends up looking decently functional. It can be interperted as well as compiled
to wasm. It is also very much a work in progress!

It looks a little like this:

```
let letter_grade(percent) = {
  if      percent >= 97 { "A+" }
  else if percent >= 93 { "A" }
  else if percent >= 90 { "A- " }
  else                  { "I'm lazy." }
};
let percent(points, total) = points / total * 100;

let grade = percent(42 * 0.97, 42); // 97%
let grade = letter_grade(grade);
print(grade); // A+
```

For more examples, check out the `./turnt/` directory.

## Building

Install the latest version of rust, and run `cargo build`.

## Running

To run a qua file, pass it as an argument. `cargo run -- /path/to/file.qua`.

To generate a wasm file, run `cargo run -- --wasm /path/to/file.qua`. The file
will be written to `../wasm-runner/file.qua.wasm`. (so make sure the directory
exists. It also expects some stdlib functions to be available. I'll publish the
code I use for this soon.)

## Tests

The (integration) tests are very very simple, just qua files along with an
associated `.out` file with the expected stdout output.

The most convenient way to run them is with [Turnt]:
`turnt ./turnt/* --parallel`.

[Turnt]: https://github.com/cucapra/turnt
