"""Gen-1 map ids.

Map ids are game data rather than WRAM, so Yellow shares Red/Blue's table --
confirmed by walking this cartridge: a new game starts on 38, the stairs lead
to 37, the front door to 0, and the lab door to 40.
"""

PALLET_TOWN = 0
REDS_HOUSE_1F = 37
REDS_HOUSE_2F = 38
OAKS_LAB = 40

NAMES = {
    PALLET_TOWN: "Pallet Town",
    REDS_HOUSE_1F: "player's house, ground floor",
    REDS_HOUSE_2F: "player's bedroom",
    OAKS_LAB: "Oak's lab",
}

# The route out, as discovered by seeded search from the bootstrap state.
INTRO_ROUTE = (REDS_HOUSE_2F, REDS_HOUSE_1F, PALLET_TOWN, OAKS_LAB)


def name(map_id: int) -> str:
    return NAMES.get(map_id, f"map {map_id}")
