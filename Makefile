.PHONY: mc ana
# light curve
LC?=Bonsai
# Dark noise rate [kHz]
darknoise:=0 5 10 20
# Dead time window [ns]
T_D:= 10 50 100 300 500 700 900
# expcted photon number
mus:= 0.1 0.5 1 5 10 20
mc: $(foreach dn,$(darknoise),$(foreach td,$(T_D),$(foreach mu,$(mus),MC/$(LC)/TD$(td)/MU$(mu)_DN$(dn).h5)))
ana_test: $(foreach dn,$(darknoise),$(foreach mu,$(mus),ANA/$(LC)/TD900/MU$(mu)_DN$(dn).h5))

MC/%.h5:
	# TD?/DN? parsed in the ToyMC.py
	# simuate nonparalyzable and paralyzable simutaneously
	mkdir -p $(@D)
	python3 ToyMC.py -o $@ --parser $(subst /,_,$*)
ANA/%.h5: MC/%.h5
	mkdir -p $(@D)
	python3 Ana.py -i $^ -o $@
ANA/$(LC)/compare.h5: $(foreach dn,$(darknoise),$(foreach td,$(T_D),$(foreach mu,$(mus),ANA/$(LC)/TD$(td)/MU$(mu)_DN$(dn).h5)))
	python3 Compare.py -i $^ -o $@
.DELETE_ON_ERRORS:
