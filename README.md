# GraphCut

Run the GraphCut-based texture synthesis program by:

```shell
python main.py [path_to_pattern] [height_of_canvas] [width_of_canvas] [mode]
```


`path_to_pattern` defaults to `data/green.gif`
`height_of_canvas` defaults to 2 times of the height of pattern
`width_of_canvas` defaults to 2 times of the width of pattern
`mode` defaults to "Matching sub-patches row by row "

For example:
```shell
python main.py data/green.gif 300 300 3
```
