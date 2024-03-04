Experiments with [tch-rs](https://github.com/LaurentMazare/tch-rs).

Ideas

- [x] GPT 2
    - [x] Training
    - [x] Inference
- [ ] Benchmarking
    - Time everything
    - Track perplexity (`e^cross_entropy_loss`)
    - Make some graphs? maybe plotnine?
    - Track tensor allocations?
- [ ] LLaMa 2
    - [ ] Load weights
    - [ ] Finetuning

## miniGPT shakespeare

Sample after 5 batches:

```
iD he, harir,
T, td hathfoX w L
;, ThasFiso tr be yvend dirHong s ther frothed tT ss e s yoll
Th?hem y atwe thEB tV&r.


Thingemave.
r d hast hosseroumou themr wW.
.
Tml mat pM te ot y t sthit in,I wnghe bN se tSattatF beistito CV xWd, acSo y tgT?schou wg gfathave imyou y heGMo
unldth y b thave pyeSitte gher be uyy ho ll
N d indFl-d sCT giorshgu I f.
That'cA J fin Fd N ou M Xlin, bowenthathI reszer, t $Ry pis w rue f cd he n,hat  ngonS merd banore c;d d thatathAy hathahaC:
APlat itinstoun d .
```

Sample after 1600 batches:

```
DERBY:
No. God morrows news, and breathe only to us,
To excuse a witing in little of his loyal
But the seasons and labour.
How their most the souls is not then.

Second Citizen:
Live them for me and smile any pace; for the
good from; repety and should right very face.

Shepherd:
I'll disschonour'd, for my kind. Wherefore my enemity
grievant us?

KING RICHARD III:
Why, I say you? Pray, sir not?

Pray:
Put the comiss and 'Reward and fashion tongues?
```

## Scuffed macOS instructions

> [!CAUTION]
> [These](https://github.com/LaurentMazare/tch-rs/issues/488#issuecomment-1825404820) instructions are to be used if and only if you understand the commands before running. It's probably way easier to just use conda/pipenv. Tested with PyTorch 2.2.0 (the current version of tch-rs doesn't support 2.2.1).

```bash
export LIBTORCH_USE_PYTORCH=1
# the brew one I couldn't get working
python3 -m pip install torch==2.2.0 --break-system-packages
# linking sucks
sudo cp /opt/homebrew/lib/python3.12/site-packages/torch/lib/* /usr/local/lib/
brew install libomp
# no clue why this couldn't be found?
sudo cp /opt/homebrew/Cellar/libomp/17.0.6/lib/libomp.dylib /usr/local/lib/
# now this should run fine
cargo run
```
