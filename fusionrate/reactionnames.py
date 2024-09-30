from collections import Counter
from enum import Enum
import re

Particles = Enum(
    "Particles",
    [
        "n",
        "H1",
        "H2",
        "H3",
        "He3",
        "He4",
        "Li6",
        "Li7",
        "Be7",
        "B11",
    ],
)

RX_SEPARATOR = '→'

DT_NAME = "T(d,n)⁴He"
DHE3_NAME = "³He(d,p)⁴He"
DDT_NAME = "D(d,p)T"
DDHE3_NAME = "D(d,n)³He"

PLI6_NAME = "⁶Li(p,h)⁴He"
PB11_NAME = "¹¹B(p,α)2⁴He"

TT_NAME = "T(t,2n)⁴He"
HT_NAME = "³He(t,pn)⁴He"
HTD_NAME = "³He(t,d)⁴He"
HH_NAME = "³He(h,2p)⁴He"
DLI6A_NAME = "⁶Li(d,α)⁴He"
DLI6N_NAME = "⁶Li(d,n)⁷Be"
DLI6P_NAME = "⁶Li(d,p)⁷Li"

ALL_REACTIONS = [
    DT_NAME,
    DHE3_NAME,
    DDT_NAME,
    DDHE3_NAME,
    PLI6_NAME,
#    PB11_NAME,
    TT_NAME,
    HT_NAME,
    HTD_NAME,
    HH_NAME,
    DLI6A_NAME,
    DLI6N_NAME,
    DLI6P_NAME,
]
# missing:
# 'D + Li-6 --> He-3 + He-4 + n'
# 'He-3 + T --> He-4 + p + n'

# no 3He because that would be ambiguous with 3 He-4
ION_SYNONYMS = {
    Particles.n: ("n-1", "n"),
    Particles.H1: ("H-1", "¹H", "H", "p"),
    Particles.H2: ("H-2", "²H", "D", "d"),
    Particles.H3: ("H-3", "³H", "T", "t"),
    Particles.He3: ("He-3", "³He", "h"),
    Particles.He4: ("He-4", "⁴He", "a", "α"),
    Particles.Li6: ("Li-6", "⁶Li", "6Li"),
    Particles.Li7: ("Li-7", "⁷Li", "7Li"),
    Particles.Be7: ("Be-7", "⁷Be", "7Be", "Be"),
    Particles.B11: ("B-11", "¹¹B", "11B", "B"),
}

PARTICLE_LOOKUP = dict()
for k, v in ION_SYNONYMS.items():
    for result in v:
        PARTICLE_LOOKUP[result] = k


def _determine_particle(s: str):
    p = PARTICLE_LOOKUP.get(s)
    if p is None:
        raise ValueError(
            f"""
            The particle '{s}' was not found in the lookup table.
            Valid particles are {sorted(set(PARTICLE_LOOKUP.keys()))}."""
        )
    return p


def _bag(*args):
    """Immutable count of the objects used as the arguments.

    Examples
    --------
    >>> _bag('a', 'a', 'b')
    frozenset({('a', 2), ('b', 1)})
    """
    counter = Counter(args)
    return frozenset(counter.items())


# unambiguous
DT_REACTANTS = _bag(Particles.H2, Particles.H3)
DHE3_REACTANTS = _bag(Particles.H2, Particles.He3)
PLI6_REACTANTS = _bag(Particles.H1, Particles.Li6)
PB11_REACTANTS = _bag(Particles.H1, Particles.B11)
TT_REACTANTS = _bag(Particles.H3, Particles.H3)

# ambiguous
DD_REACTANTS = _bag(Particles.H2, Particles.H2)
HH_REACTANTS = _bag(Particles.He3, Particles.He3)
HT_REACTANTS = _bag(Particles.H3, Particles.He3)
DLI6_REACTANTS = _bag(Particles.H2, Particles.Li6)
HLI6_REACTANTS = _bag(Particles.He3, Particles.Li6)

_REACTIONS = {
    (DT_REACTANTS, _bag(Particles.n, Particles.He4)): DT_NAME,
    (DHE3_REACTANTS, _bag(Particles.H1, Particles.He4)): DHE3_NAME,
    (PLI6_REACTANTS, _bag(Particles.He3, Particles.He4)): PLI6_NAME,
    (
        PB11_REACTANTS,
        _bag(Particles.He4, Particles.He4, Particles.He4),
    ): PB11_NAME,
    (TT_REACTANTS, _bag(Particles.He4, Particles.n, Particles.n)): TT_NAME,
    (HH_REACTANTS, _bag(Particles.He4, Particles.H1, Particles.H1)): HH_NAME,
    (DD_REACTANTS, _bag(Particles.H3, Particles.H1)): DDT_NAME,
    (DD_REACTANTS, _bag(Particles.He3, Particles.n)): DDHE3_NAME,
    (HT_REACTANTS, _bag(Particles.He4, Particles.n, Particles.H1)): HT_NAME,
    (HT_REACTANTS, _bag(Particles.He4, Particles.H2)): HTD_NAME,
    (DLI6_REACTANTS, _bag(Particles.He4, Particles.He4)): DLI6A_NAME,
    (DLI6_REACTANTS, _bag(Particles.n, Particles.Be7)): DLI6N_NAME,
    (DLI6_REACTANTS, _bag(Particles.H1, Particles.Li7)): DLI6P_NAME,
}


def _generate_single_branch_list(reactions_lookup: dict):
    """Get reactions that have only have one possible set of products"""
    all_reactions = list(reactions_lookup.keys())
    all_reactants = [r[0] for r in all_reactions]
    with_unique_reactants = {}
    for r in all_reactions:
        reactants, products = r
        if all_reactants.count(reactants) == 1:
            with_unique_reactants[reactants] = reactions_lookup[r]
    return with_unique_reactants


_REACTIONS_WITH_UNIQUE_PRODUCTS = _generate_single_branch_list(_REACTIONS)


def _to_particle(s: str):
    s = s.strip()
    particle = PARTICLE_LOOKUP.get(s)
    if particle is None:
        raise ValueError(
            f"""
            The particle '{s}' was not found in the lookup table.
            Valid particles are {sorted(set(PARTICLE_LOOKUP.keys()))}."""
        )
    return particle


def _normalize_reaction_separators(s: str):
    s = re.sub(r"-+>", RX_SEPARATOR, s)
    s = re.sub(r",", RX_SEPARATOR, s)
    return s


def _count_reaction_separators(s: str):
    s = _normalize_reaction_separators(s)
    return s.count(RX_SEPARATOR)


def _joinparticles(particles: list):
    return "+".join(particles)


def _splitparticles(s: str):
    return re.split(r'\(|\)|\+', s)


def _validate_reaction_string(s: str):
    number_of_separators = _count_reaction_separators(s)
    if number_of_separators not in (0, 1):
        raise ValueError(
            f"""Reaction string '{s}'
           has more than one reactant -> product separator.
           Only one instance of {RX_SEPARATOR} or -> is allowed."""
        )


def _multiply_particles(reaction_string):
    """Turn '2 ³He' to '³He + ³He'"""
    # This pattern is a bit of a hack.
    # It purposefully places 'H' at the end since it's a prefix of many of the
    # other strings. Other than that, only 'n' is a prefix of 'n-1'.
    s = reaction_string.strip()
    searchable_particles = [
        "n-1",
        "n",
        "p",
        "α",
        "h",
        "D",
        "T",
        "He-3",
        "³He",
        "He-4",
        "⁴He",
        "H-1",
        "H-2",
        "H-3",
        "H",
    ]
    pattern = (
        r"(?P<coeff>2|3)\s*(?P<particle>"
        + "|".join(searchable_particles)
        + ")"
    )
    matches = re.match(pattern, s)
    if matches:
        coeff = int(matches.group("coeff"))
        particle = matches.group("particle")
        output_components = [particle] * coeff
        return _joinparticles(output_components)
    else:
        # If the input string doesn't match the pattern, return original string
        return s


def _expand_particle_description(s: str):
    s1 = _splitparticles(s)
    s2 = [_multiply_particles(s) for s in s1]
    s3 = _joinparticles(s2)
    s4 = re.sub(r"pn|np", "p+n", s3)
    return s4


def _parse_reactants(s: str):
    s_expanded = _expand_particle_description(s)
    particles = _splitparticles(s_expanded)
    if len(particles) != 2:
        raise ValueError(
            f"""The reactant description '{s}' must specify exactly two
            particles."""
        )
    return _bag(*map(_to_particle, particles))


def _parse_products(s: str):
    s_expanded = _expand_particle_description(s)
    particles = _splitparticles(s_expanded)
    num_particles = len(particles)
    if num_particles < 2 or num_particles > 3:
        raise ValueError(
            f"""The product description '{s}' must specify
            either two or three particles. {num_particles} were detected."""
        )
    return _bag(*map(_to_particle, particles))


def target_species(s: str):
    r"""Name of reaction target species
    Parameters
    ----------
    s: string
        canonical reaction name

    Returns
    -------
    Canonical non-charge-specific species form

    Examples
    --------
    >>> reaction_target_species("T(d,n)⁴He")
    'T'

    >>> reaction_target_species("³He(t,pn)⁴He")
    '³He'
    """
    return s.split("(")[0]


def particle_form_to_target_form(s: str):
    r"""For supported species only

    These species appear in the canonical reaction names.
    Here we convert to 'uppercase' species forms.

    Parameters
    ----------
    s : {'p', 'd', 't', 'h'}
        lowercase species form

    Returns
    -------
    Canonical non-charge-specific species form

    Examples
    --------
    >>> particle_form_to_target_form('p')
    'H'

    >>> particle_form_to_target_form('h')
    '³He'
    """
    d = {"p": "H", "d": "D", "t": "T", "h": "³He"}
    return d[s]


def beam_species(s: str):
    r"""Name of reaction beam species

    Parameters
    ----------
    s: string
        canonical reaction name

    Returns
    -------
    Canonical non-charge-specific species form

    Examples
    --------
    >>> reaction_beam_species("T(d,n)⁴He")
    'D'

    >>> reaction_beam_species("³He(t,pn)⁴He")
    'T'
    """
    reactants = s.split(",")[0]
    beam_sp = reactants.split("(")[1]
    return particle_form_to_target_form(beam_sp)


def reactants(s: str):
    r"""Names of beam, target species

    Parameters
    ----------
    s: string
        canonical reaction name

    Returns
    -------
    Tuple of particles in canonical non-charge-specific species form

    Examples
    --------
    >>> reactants("T(d,n)⁴He")
    'D', 'T'

    >>> reactants("³He(t,pn)⁴He")
    'T', '³He'
    """
    beam = beam_species(s)
    target = target_species(s)
    return beam, target


def _name_parser(s: str):
    count = _count_reaction_separators(s)
    if count > 1:
        raise ValueError(
            f"""Reaction string '{s}'
           has more than one reactant -> product separator.
           Only one instance of {RX_SEPARATOR} or -> or , is allowed."""
        )

    if count == 0:
        reactants = _parse_reactants(s)
        canonical_name = _REACTIONS_WITH_UNIQUE_PRODUCTS.get(reactants)

    else:
        s = _normalize_reaction_separators(s)
        reactants, products = s.split(RX_SEPARATOR)

        reactants_description = _parse_reactants(reactants)
        products_description = _parse_products(products)
        reaction_description = (reactants_description, products_description)

        canonical_name = _REACTIONS.get(reaction_description)

    return canonical_name


def _extra_name_resolver(reaction_raw_name: str):
    r"""Recognize a canonical fusion reaction

    The name resolver aims to recognize the common fusion reactions
    from a string, in a reasonable number of unambiguous formats.

    Parameters
    ----------
    reaction_raw_name: string
        The name resolver aims to recognize the common fusion reactions
        from a string, in a reasonable number of unambiguous formats.

    Examples
    --------
    >>> name_resolver("DT")
    'T(d,n)⁴He'

    >>> name_resolver("D+3He")
    '³He(d,p)T'
    """
    resolvers = [
            bosch_name_resolver,
            proton_boron_name_resolver,
            proton_lithium_name_resolver
            ]

    for resolver in resolvers:
        try:
            return resolver(reaction_raw_name)
        except ValueError:
            continue

    return None

def name_resolver(s: str):
    r"""Recognize a canonical fusion reaction

    The name resolver aims to recognize the common fusion reactions
    from a string, in a reasonable number of unambiguous formats.

    Parameters
    ----------
    reaction_raw_name: string
        The name resolver aims to recognize the common fusion reactions
        from a string, in a reasonable number of unambiguous formats.

    Examples
    --------
    >>> name_resolver("DT")
    'T(d,n)⁴He'

    >>> name_resolver("D+T→α+n")
    'T(d,n)⁴He'

    >>> name_resolver("D + He-3")
    '³He(d,p)T'
    """
    if s in ALL_REACTIONS:
        return s

    canonical_name = _extra_name_resolver(s)
    if canonical_name is not None:
        return canonical_name

    canonical_name = _name_parser(s)
    if canonical_name is not None:
        return canonical_name

    raise ValueError(
        f"""
    The reaction name resolver could not determine the reaction
    {reaction_raw_name}. The best options are {ALL_REACTIONS}.
    """
    )


def reaction_name_simplify(reaction_name_raw: str):
    """Convert special characters to a standard form for matching"""
    s = reaction_name_raw.replace(" ", "")
    s = s.replace("¹", "1")
    s = s.replace("²", "2")
    s = s.replace("³", "3")
    s = s.replace("⁴", "4")
    s = s.replace("⁶", "6")
    s = s.replace("⁷", "7")
    s = s.replace("->", "→")
    s = s.replace("-", "+")
    s = s.replace("α", "a")
    return s


def reaction_name_to_endf(canonical_reaction_name: str):
    s = canonical_reaction_name
    s = s.replace("4He", "a")
    s = s.replace("3He", "h")
    s = s.replace("T", "t")
    s = s.replace("D", "d")
    return s


def reaction_filename_part(canonical_reaction_name):
    s = reaction_name_simplify(canonical_reaction_name)
    s = reaction_name_to_endf(s)
    return s


def proton_boron_name_resolver(reaction_raw_name):
    NAMES = [
        PB11_NAME,
        "pB",
        "pB11",
    ]

    reaction_name = reaction_name_simplify(reaction_raw_name)

    for n in NAMES:
        if reaction_name == reaction_name_simplify(n):
            return PB11_NAME

    raise ValueError(
        f"""
        In the proton_boron_name_resolver, {reaction_raw_name} could not be
        resolved. Possible options are {NAMES}.
        """
    )


def proton_lithium_name_resolver(reaction_raw_name):
    NAMES = [
        "pLi6",
    ]

    reaction_name = reaction_name_simplify(reaction_raw_name)

    for n in NAMES:
        if reaction_name == reaction_name_simplify(n):
            return PLI6_NAME

    raise ValueError(
        f"""
        In the proton_lithium_name_resolver, {reaction_raw_name} could not be
        resolved. Possible options are {NAMES}.
        """
    )


# this could be re-worked to use a dict so that you don't need to loop over all
# the entries in sequence.
def bosch_name_resolver(reaction_raw_name: str):
    DT_NAMES = [DT_NAME, "DT"]

    DHE3_NAMES = [
        DHE3_NAME,
        "DHe",
        "D3He",
        "D+3He",
        "DHe3",
    ]

    DDHE3_NAMES = [
        DDHE3_NAME,
        "D(d,n)3He",
        "D+D→n+3He",
        "D+D→3He+n",
        "²H+²H→n+3He",
        "²H+²H→3He+n",
    ]

    reaction_name = reaction_name_simplify(reaction_raw_name)

    for name_collection in [DT_NAMES, DHE3_NAMES, DDHE3_NAMES]:
        for n in name_collection:
            if reaction_name == reaction_name_simplify(n):
                return name_collection[0]

    raise ValueError(
        f"""
        In the bosch_name_resolver, {reaction_raw_name} could not be
        resolved. Possible options are {DT_NAMES}, {DHE3_NAMES},
        {DDHE3_NAMES}."""
    )


if __name__ == "__main__":
    print(name_resolver("DT"))
