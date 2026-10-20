from poke.maps import INTRO_ROUTE, NAMES, OAKS_LAB, PALLET_TOWN, REDS_HOUSE_2F, name


def test_route_out_matches_what_the_search_walked():
    """Bedroom -> ground floor -> Pallet Town -> lab, as found by seeded search
    on the cartridge."""
    assert INTRO_ROUTE == (38, 37, 0, 40)
    assert INTRO_ROUTE[0] == REDS_HOUSE_2F
    assert INTRO_ROUTE[-1] == OAKS_LAB


def test_every_route_map_is_named():
    assert all(map_id in NAMES for map_id in INTRO_ROUTE)


def test_unknown_map_falls_back_to_its_number():
    assert name(PALLET_TOWN) == "Pallet Town"
    assert name(231) == "map 231"
