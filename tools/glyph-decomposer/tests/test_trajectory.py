from glyph_decomposer.trajectory import extend_directed_route, teacher_forced_routes


def test_inertial_route_continues_straight_through_a_junction():
    skeleton = {(x, 5) for x in range(1, 10)} | {(5, y) for y in range(5, 9)}
    route = [[1, 5], [2, 5], [3, 5], [4, 5], [5, 5]]

    completed, evidence = extend_directed_route(route, skeleton)

    assert completed[0] == [1, 5]
    assert completed[-1] == [9, 5]
    assert evidence == {"startAdded": 0, "endAdded": 4}


def test_inertial_route_extends_both_missing_ends():
    skeleton = {(x, 5) for x in range(1, 10)}
    route = [[4, 5], [5, 5], [6, 5]]

    completed, evidence = extend_directed_route(route, skeleton)

    assert completed[0] == [1, 5]
    assert completed[-1] == [9, 5]
    assert evidence == {"startAdded": 3, "endAdded": 3}


def test_straight_stroke_does_not_turn_into_another_branch():
    skeleton = {(5, y) for y in range(1, 8)} | {
        (6, 8),
        (7, 9),
        (8, 10),
        (9, 11),
    }
    route = [[5, 1], [5, 2], [5, 3], [5, 4], [5, 5]]

    completed, _evidence = extend_directed_route(route, skeleton, "竖")

    assert completed[-1][0] <= 6
    assert [9, 11] not in completed


def test_teacher_route_uses_landmark_to_choose_branch():
    skeleton = [[x, 5] for x in range(1, 10)] + [[5, y] for y in range(6, 10)]
    payload = {
        "size": 100,
        "skeleton": skeleton,
        "truth": [[[1, 5], [5, 5], [5, 9]]],
    }

    routes = teacher_forced_routes(payload)

    assert routes == [
        [[1, 5], [2, 5], [3, 5], [4, 5], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9]]
    ]
