# lyrics-generator
lyrics-generator is a small experiment in generating lyrics with Long short-term memory (LSTM) networks.<br/>
The experiments with LSTM models were conducted on an Serbian lyrics dataset.

## Dataset
The dataset we used to generate training data: https://tekstovi.net/2,0,0.html

## Models we trained (kinda)
- lstm-512-4 (512 units, 4 layers) train loss: 1.56, dev loss:  1.6
- here should be more models
- but we are poor and
- we can't train because we don't have hardware (besides that, we are lazy) :cry:

## Results (only people from Balkan will understand this, or not even them) :grimacing:
Some examples of lstm-512-4 evaluation 
```
ja sam preboleo
srecna nije samo jedan stan
kad se nisam ne znam
sad mi drugi se vrata
pa sve mi noci dvije
nad zoro dosta mala mi najmali
samo druzi nas dana sam
sve mi davom samo tebe
nijedna nikad ne znam
------------------------
ja sam sa mnom
ne znam kao da nocas ponovo
prazin te nisam znao
da mi se pozivim
s tuznum se vino moj
kad si me starila
sve sam polud znao
nista ne boli, to, da patim ja
svima tebe nista novo
nije mi sve
------------------------
hajde i srce moje
kad sam ti duso daleko
da se zovam da se vrate
puta mi je bez tebe
necu znam ti sad mi se vole
nije vise kada ne vrata
pogledaj sam ti da me sada stida
sve sto ti nesto divno starimo stara
```
