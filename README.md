# Getting started (the best practice)

1. Follow instruction on https://julialang.org/downloads/ and get the latest release of Julia
    - For Linux/macOS, `curl -fsSL https://install.julialang.org | sh` gets the job done
2. clone this repo to a location you like
3. Run Julia executable, you will be dropped into a REPL:
```
$ > julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.11.6 (2025-07-09)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia>
```
5. hit `;` once to switch to shell mode, cd to where you cloned this repo (you can cd before running
   `julia` in terminal, same effect)
6. hit `]` once to switch to Pkg mode, and run
```
pkg> activate .
```
where the `dot` means here.

7. run `pkg> instantiate`
8. Look up again when waiting for precompilation

## Compiled Pluto notebooks:
- [Fitting in Julia](https://pluto.land/n/m6fuf72z)
- [MLJ Demo](https://pluto.land/n/qbhxq78c)
