The masses in isotopes.json were generated using Mathematica 13.0.

```
isomass[x_] :=
 Entity["Isotope", x]["AtomicMass"] -
   Quantity[Entity["Isotope", x]["AtomicNumber"], "ElectronMass"] //
  UnitConvert[#, "AtomicMassConstant"] &


isotopes = {"Hydrogen1", "Hydrogen2", "Hydrogen3", "Helium3",
  "Helium4", "Lithium6", "Lithium7", "Beryllium7", "Boron11"}

QuantityMagnitude /@ isomass /@ isotopes
```

The neutron mass was generated similarly.
This does neglect the electronic binding energy that
would have been included in AtomicMass.
