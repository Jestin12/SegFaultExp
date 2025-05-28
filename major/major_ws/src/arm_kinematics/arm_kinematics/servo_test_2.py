

def find_PWM(angle, min_pw, max_pw, pwm_freq_hz): 
		pulse_width_ms = min_pw + (angle / 180.0) * (max_pw - min_pw)
		print(pulse_width_ms)
		period_ms = 1000 / pwm_freq_hz
		duty_cycle = (pulse_width_ms / period_ms) * 100

		return duty_cycle



while True:

    ameline = input("enter an angle:")
    angle = float(ameline)

    print(find_PWM(angle, 1.5, 2.5, 50))
    
    






