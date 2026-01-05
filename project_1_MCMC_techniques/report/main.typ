#import "template.typ": *
#import "@preview/cetz:0.4.2"
#import "@preview/muchpdf:0.1.2": muchpdf

// Algorithm
#import "@preview/algorithmic:1.0.7"
#import algorithmic: algorithm-figure, style-algorithm
#show: style-algorithm

#set text(lang: "nb")


#show: ilm.with(
  title: [Fysikk-informerte nevrale nettverk],
  author: "Prosjekt i TMA4320. Skrevet av Elling Svee.",
  // date: datetime(year: 2026, month: 01, day: 18),
  date: datetime.today(),
  // abstract: [],
  bibliography: bibliography("refs.bib", style: "elsevier-harvard"),
  figure-index: (enabled: false),
  table-index: (enabled: false),
  listing-index: (enabled: true),
  table-of-contents: none,
  chapter-pagebreak: false,
  fancy-cover-page: false,
)



#block(width: 100%)[
  #text(24pt, weight: "bold")[Fysikk-informerte nevrale nettverk]
  #v(-1.5em)
  #text(1.2em)[Prosjekt i TMA4320]
]
= Introduksjon

jadf



