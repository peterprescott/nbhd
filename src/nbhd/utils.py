def get_pixel(postcode):
    response = db.query(f"SELECT * FROM names WHERE name1 = '{postcode.upper()}'", True)
    point = response.geometry[0]
    pixels = db.intersects('pixels', point)
    pixel = pixels.iloc[0].geometry
    return pixel
