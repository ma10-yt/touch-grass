import math
from app.utils.geo import calculate_distance

def test_calculate_distance():
    # Example points: Paris and London
    lat1, lon1 = 48.8566, 2.3522
    lat2, lon2 = 51.5074, -0.1278
    
    dist = calculate_distance(lat1, lon1, lat2, lon2)
    # Expected distance is around 343 km
    assert math.isclose(dist, 343000, rel_tol=0.05)

def test_same_point():
    dist = calculate_distance(10.0, 10.0, 10.0, 10.0)
    assert dist == 0.0

