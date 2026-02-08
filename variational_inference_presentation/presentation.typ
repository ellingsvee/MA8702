#import "@preview/touying:0.6.1": *
#import "@preview/subpar:0.2.2"
#import "@preview/muchpdf:0.1.2": muchpdf
#import "@preview/herodot:0.4.0": *

#import themes.simple: *

#show: simple-theme.with(
  header-right: none,
  primary: rgb("#00509e"),
)

#set text(
  font: "Times New Roman",
  size: 25pt,
)


#set figure(gap: 0.0em)
#let big-text(body) = text(size: 35pt)[#body]

#let fill-color = luma(250)

#show link: set text(fill: blue)


#set raw(lang: "bash")

#set align(horizon)

#show raw.where(block: false): box.with(
  fill: fill-color.darken(5%),
  inset: (x: 3pt, y: 0pt),
  outset: (y: 5pt, x: 2pt),
  radius: 2pt,
)

// Display block code with padding.
#show raw.where(block: true): block.with(
  fill: fill-color.darken(5%),
  inset: (x: 3pt, y: 2pt),
  outset: (x: 0pt, y: 3pt),
  radius: 2pt,
  width: 100%,
)

#let resize-text(body) = layout(size => {
  let font_size = text.size
  let (height,) = measure(
    block(width: size.width, text(size: font_size)[#body]),
  )
  let max_height = size.height

  while height > max_height {
    font_size -= 0.2pt
    height = measure(
      block(width: size.width, text(size: font_size)[#body]),
    ).height
  }

  block(
    height: height,
    width: 100%,
    text(size: font_size)[#body],
  )
})


#title-slide[
  = Variational Inference:\ A Review for Statisticians

  #cite(<blei_variational_2017>, form: "prose")
]

== Test




==
#text(size: 15pt)[
  #bibliography("refs.bib", style: "elsevier-harvard")
]

