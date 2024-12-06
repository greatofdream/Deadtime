.PHONY: mc ana
# light curve: Bonsai, Rectangle
LC?=Rectangle
# Dark noise rate [kHz]
darknoise:=0 5 10 20
# Dead time window [ns]
T_D:= 10 50 100 300 500 700 900
# expcted photon number
mus:= $(shell seq 0.1 0.1 0.9) $(shell seq 1 1 10) $(shell seq 15 5 40)
mc: $(foreach dn,$(darknoise),$(foreach td,$(T_D),$(foreach mu,$(mus),MC/$(LC)/TD$(td)/MU$(mu)_DN$(dn).h5)))
ana_test: $(foreach dn,$(darknoise),$(foreach mu,$(mus),ANA/$(LC)/TD900/MU$(mu)_DN$(dn).h5))

MC/$(LC)/%.h5:
	# TD?/DN? parsed in the ToyMC.py
	# simuate nonparalyzable and paralyzable simutaneously
	mkdir -p $(@D)
	python3 ToyMC.py -o $@ --parser $(subst /,_,$*) --model $(LC)
MC/$(LC)/compare.h5:
	python3 CompareMC.py --MU $(mus) --DN $(darknoise) --TD $(T_D) --format $(@D)/TD{}/MU{}_DN{}.h5 -o $@
ANA/$(LC)/%.h5: MC/$(LC)/%.h5
	mkdir -p $(@D)
	python3 AnaTot.py -i $^ -o $@ --parser $(subst /,_,$*)
ANA/$(LC)/compare.h5: $(foreach dn,$(darknoise),$(foreach td,$(T_D),$(foreach mu,$(mus),ANA/$(LC)/TD$(td)/MU$(mu)_DN$(dn).h5)))
	python3 Compare.py -i $^ -o $@
.DELETE_ON_ERRORS:
