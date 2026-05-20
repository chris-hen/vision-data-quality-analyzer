# PlotNeuralNet Layer Macros

This folder contains the custom TikZ/LaTeX layer macros used for the CNN
architecture visualization of this project.

The visualization is based on modified PlotNeuralNet-style components.

Included files:

* `Box.sty`
* `RightBandedBox.sty`
* `Ball.sty`
* `init.tex`

The original macros were adapted and extended for this project to create
a cleaner and more presentation-oriented CNN visualization.

Main modifications include:

* custom visual styling
* modified layer appearance
* configurable face and edge visibility
* improved activation band rendering
* improved dense/output layer visualization
* compatibility with the custom dashboard visualization pipeline

These macros are used by:

`cnn_plotneuralnet_portfolio.tex`

Compilation workflow:

1. Python generates the `.tex` file
2. MiKTeX compiles the LaTeX/TikZ visualization
3. Inkscape converts the resulting PDF into SVG format
4. The SVG visualization is displayed inside the Streamlit dashboard
