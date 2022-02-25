import numpy as np

#X: [-6*2 .. 6*2),  Y: [0 .. +10). 
ctc_id_arr = np.array([[ 2, 13, 33, 43, 63, 50, 81, 68,  88,  98, 118,  65],
	[26, 12, 28, 42, 58, 47, 84, 73,  89, 103, 119, 105],
	[20, 15, 25, 45, 55, 53, 78, 76,  86, 106, 116, 111],
	[17, 10, 24, 40, 54, 44, 87, 77,  91, 107, 121, 114],
	[23,  7, 27, 37, 57, 38, 93, 74,  94, 104, 124, 108],
	[14,  6, 22, 36, 52, 35, 96, 79,  95, 109, 125, 117],
	[ 8,  9, 19, 39, 49, 41, 90, 82,  92, 112, 122, 123],
	[ 5,  4, 18, 34, 48,  1, 66, 83,  97, 113, 127, 126],
	[11, 32, 21, 31, 51, 61, 70, 80, 100, 110,  99, 120],
	[ 3, 29, 16, 30, 46, 60, 71, 85, 101, 115, 102,  64]])

#X: [6*2 .. 16*2),  Y: [0 .. +10). 
ctr_id_arr = np.array([[ 29, 16, 30, 46, 60, 81, 68, 88,  98, 118],
	[32, 21, 31, 51, 61, 84, 73, 89, 103, 119],
	[ 4, 18, 34, 48,  1, 78, 76, 86, 106, 116],
	[ 9, 19, 39, 49, 41, 87, 77, 91, 107, 121],
	[ 6, 22, 36, 52, 35, 93, 74, 94, 104, 124],
	[ 7, 27, 37, 57, 38, 96, 79, 95, 109, 125],
	[10, 24, 40, 54, 44, 90, 82, 92, 112, 122],
	[15, 25, 45, 55, 53, 66, 83, 97, 113, 127],
	[12, 28, 42, 58, 47, 70, 80, 100, 110, 99],
	[13, 33, 43, 63, 50, 71, 85, 101, 115, 102]])

#X: (-16*2 .. -6*2],  Y: (+10 .. 0].
ctl_id_arr = np.array([[13, 33, 43, 63, 50, 71, 85, 101, 115, 102],
	[12, 28, 42, 58, 47, 70, 80, 100, 110, 99],
	[15, 25, 45, 55, 53, 66, 83,  97, 113, 127],
	[10, 24, 40, 54, 44, 90, 82,  92, 112, 122],
	[ 7, 27, 37, 57, 38, 96, 79,  95, 109, 125],
	[ 6, 22, 36, 52, 35, 93, 74,  94, 104, 124],
	[ 9, 19, 39, 49, 41, 87, 77,  91, 107, 121],
	[ 4, 18, 34, 48,  1, 78, 76,  86, 106, 116],
	[32, 21, 31, 51, 61, 84, 73,  89, 103, 119],
	[29, 16, 30, 46, 60, 81, 68,  88,  98, 118]])

#X: [0 .. 16*4),  Y: (20 .. 0]. 
str_id_arr = np.array([[11, 14, 32,  6, 21, 22, 31, 36, 51, 61, 84, 73,  89, 103, 119, 105],
					  [ 3, 20, 29, 15, 16, 25, 30, 45, 46, 60, 81, 68,  88,  98, 118,  65],
					  [ 8, 17,  9, 10, 19, 24, 39, 40, 49, 41, 87, 77,  91, 107, 121, 114],
					  [ 5,  2,  4, 13, 18, 33, 34, 43, 48,  1, 78, 76,  86, 106, 116, 111],
					  [23, 26,  7, 12, 27, 28, 37, 42, 57, 38, 96, 79,  95, 109, 125, 117],
					  [ 0,  0,  0,  0,  0,  0,  0,  0, 52, 35, 93, 74,  94, 104, 124, 108],
					  [ 0,  0,  0,  0,  0,  0,  0,  0, 55, 53, 66, 83,  97, 113, 127, 126],
					  [ 0,  0,  0,  0,  0,  0,  0,  0, 54, 44, 90, 82,  92, 112, 122, 123],
					  [ 0,  0,  0,  0,  0,  0,  0,  0, 63, 50, 71, 85, 101, 115, 102,  64],
                      [ 0,  0,  0,  0,  0,  0,  0,  0, 58, 47, 70, 80, 100, 110,  99, 120]])

#X: (-16*4 .. 0], Y: (20 .. 0]. 
stl_id_arr = np.array([[26, 12, 28, 42, 58, 47, 70, 80, 95, 100, 109, 110, 125,  99, 117, 120],
					 [ 2, 13, 33, 43, 63, 50, 71, 85, 86, 101, 106, 115, 116, 102, 111,  64],
					 [17, 10, 24, 40, 54, 44, 90, 82, 91,  92, 107, 112, 121, 122, 114, 123],
					 [20, 15, 25, 45, 55, 53, 66, 83, 88,  97,  98, 113, 118, 127,  65, 126],
					 [14,  6, 22, 36, 52, 35, 93, 74, 89,  94, 103, 104, 119, 124, 105, 108],
					 [23,  7, 27, 37, 57, 38, 96, 79,  0,   0,   0,   0,   0,   0,   0,   0],
					 [ 5,  4, 18, 34, 48,  1, 78, 76,  0,   0,   0,   0,   0,   0,   0,   0],
					 [ 8,  9, 19, 39, 49, 41, 87, 77,  0,   0,   0,   0,   0,   0,   0,   0],
					 [ 3, 29, 16, 30, 46, 60, 81, 68,  0,   0,   0,   0,   0,   0,   0,   0],
					 [11, 32, 21, 31, 51, 61, 84, 73,  0,   0,   0,   0,   0,   0,   0,   0]])

# zone_id = {'stl' : 2, 'str' : -1,
#            'ctl' : -1, 'ctc' : -1, 'ctr' : -1,
#            'cbl' : -1, 'cbc' : -1, 'cbr' : -1,
#            'sbl' : 7, 'sbr' : -1}
#Mapping for 2 readout zones May 2021

# zone_id = {'stl' : -1, 'str' : -1,
#           'ctl' : -1, 'ctc' : 2, 'ctr' : -1,
#           'cbl' : -1, 'cbc' : 7, 'cbr' : -1,
#           'sbl' : -1, 'sbr' : -1}

#Mapping for 6 readout zones started in June 2021
#zone_id = {'stl' : -1, 'str' : -1,
#            'ctl' : 3, 'ctc' : 1, 'ctr' : 0,
#            'cbl' : 7, 'cbc' : 6, 'cbr' : 5,
#            'sbl' : 8, 'sbr' : -1} 


#Mapping
#zone_id = {'stl' : -1, 'str' : -1,
#            'ctl' : -3, 'ctc' : 1, 'ctr' : -1,
#            'cbl' : -7, 'cbc' : 6, 'cbr' : -5,
#            'sbl' : -1, 'sbr' : -1} 

#10 June Remove sbl (8) board from mapping
# zone_id = {'stl' : -1, 'str' : -1,
#            'ctl' : 3, 'ctc' : 1, 'ctr' : 0,
#             'cbl' : 7, 'cbc' : 6, 'cbr' : 5,
#             'sbl' : -1, 'sbr' : -1} 

def get_side_ch_id(x,y, zone_id):
	if x < 16 and y < 10: # x:[0,15] y:[0,9]
		ch = str_id_arr[y, 15-x]
		zone = 'sbl'
	elif x >= 16 and y < 10: # x:[16,31] y:[0,9]
		ch = stl_id_arr[y, 31-x]
		zone = 'sbr'
	elif x < 16 and y >= 10: # x:[0,15] y:[0,9]
		ch = stl_id_arr[19-y, x]
		zone = 'stl'
	else: # x:[16,31] y:[10,19]
		ch = str_id_arr[19-y, x-16]
		zone = 'str'
	return (zone_id.get(zone)*128 + ch)*(zone_id.get(zone)>=0)

def get_center_ch_id(x,y, zone_id):
	if y < 10:
		y = 9 - y
		if x < 10:
			zone = 'cbl'
			ch = ctl_id_arr[y, 9-x]
		elif x > 21:
			zone = 'cbr'
			ch = ctr_id_arr[y, 21-x]
		else:
			zone = 'cbc' 
			ch = ctc_id_arr[9-y,9-x]
	else:
		y = 19 - y 
		if x < 10:
			zone = 'ctl'
			ch = ctl_id_arr[y, x]
		elif x > 21:
			zone = 'ctr'
			ch = ctr_id_arr[y, x-22]
		else:
			zone = 'ctc' 
			ch = ctc_id_arr[y, x-10]  

	return (zone_id.get(zone)*128 + ch)*(zone_id.get(zone)>=0)

def get_xy(ch, zone_id):
    zone_num = int(ch/128)
    zone_pos = [key for key, value in zone_id.items() if value == zone_num]
    raw_ch = int(ch)%128
    #print(f'zone: {zone_num:1.0f}, raw_ch: {raw_ch: 4.0f}')
    if zone_pos:
        if zone_pos[0] == 'cbl':
            raw_y, raw_x = np.where(ctl_id_arr==raw_ch)
            x = 9-raw_x
            y = 9-raw_y
        elif  zone_pos[0] == 'cbr':
            raw_y, raw_x = np.where(ctr_id_arr==raw_ch)
            x = 31 - raw_x
            y = 9 - raw_y
        elif zone_pos[0] == 'cbc':
            raw_y, raw_x = np.where(ctc_id_arr==raw_ch)
            x = 21 - raw_x
            y = raw_y #TODO: understand y difference between central and side plates
        elif zone_pos[0] == 'ctl':
            raw_y, raw_x = np.where(ctl_id_arr==raw_ch)
            x = raw_x
            y = 19 - raw_y
        elif  zone_pos[0] == 'ctr':
            raw_y, raw_x = np.where(ctr_id_arr==raw_ch)
            x = 22 + raw_x
            y = 19 - raw_y
        elif zone_pos[0] == 'ctc':
            raw_y, raw_x = np.where(ctc_id_arr==raw_ch)
            x = 10 + raw_x
            y = 19 - raw_y  
        if not np.shape(raw_x)[0]:
            #print('Unable to find this raw channel at channel map')
            #print(f'zone: {zone_num:1.0f}, raw_ch: {raw_ch: 4.0f}')
            x = np.array([-1])
            y = np.array([-1])
    else:
        #print('Event does not match to the detector sensitive area')
        #print(f'zone: {zone_num:1.0f}, raw_ch: {raw_ch: 4.0f}')
        x = np.array([-1])
        y = np.array([-1])
    return [x[0],y[0]]
