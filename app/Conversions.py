class Conversions():
    def __init__(self):
        pass

    def meter_feet(self, length, conv_to):
        if conv_to == "meter":
            meter = length * 0.3048
            return meter
        elif conv_to == "ft":
            feet = length / 0.3048
            return feet
    
    def meter_inch(self, length, conv_to):
        if conv_to == "meter":
            meter = length / 39.37008
            return meter
        elif conv_to == "inch":
            inch = length * 39.37008
            return inch

    def m2_ft2(self, area, conv_to):
        if conv_to == "ft2":
            ft2 = area * 10.76391042
            return ft2
        elif conv_to == "m2":
            m2 = area / 10.76391042
            return m2
    
    def m1s_kn(self, vel, conv_to):
        if conv_to == "kn":
            kn = vel / 0.5144
            return kn
        elif conv_to == "m1s":
            m1s = vel * 0.5144
            return m1s

    def km1h_m1s(self, vel, conv_to):
        if conv_to == "km1h":
            km1h = vel * 3.6
            return km1h
        elif conv_to == "m1s":
            m1s = vel / 3.6
            return m1s

    def km1h_kn(self, vel, conv_to):
        if conv_to == "kn":
            kn = vel * 0.540003
            return kn
        elif conv_to == "km1h":
            km1h = vel / 0.540003
            return km1h

    def kn_mph(self, vel, conv_to):
        if conv_to == "mph":
            mph = vel * 1.150779448
            return mph
        elif conv_to == "kn":
            kn = vel / 1.150779448
            return kn

    def m1s_ft1s(self, vel, conv_to):
        if conv_to == "ft1s":
            ft1s = vel * 3.28084
            return ft1s
        elif conv_to == "m1s":
            m1s = vel / 3.28084
            return m1s
        
    def kg_pound(self, mass, conv_to):
        if conv_to == "kg":
            kg = mass * 0.45359237
            return kg
        elif conv_to == "pound":
            lbs = mass / 0.45359237
            return lbs
    
    def pa_psf(self, pressure, conv_to):
        if conv_to == "psf":
            psf = pressure / 47.88025951
            return psf
        elif conv_to == "pa":
            pa = pressure * 47.88025951
            return pa