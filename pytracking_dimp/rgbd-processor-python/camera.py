import numpy as np
'''
getCameraParam: Get the camera matrix C
Input: choose between color camera or depth camera
Output: Intrinsic camera matrix
'''
def processCamMat(strList):
    if(len(strList) == 1):
        numbers = np.array([strList[0].split(' ')[:9]]).astype(float)
        mat = np.reshape(numbers,[3,3], 'C')
    else:
        mat = np.zeros([3,3])
        for idx, line in enumerate(strList):
            line = line.rstrip() #rstrip() returns a copy of the string in which all chars have been stripped from the end of the string (default whitespace characters).
            numbers = line.split(' ')
            mat[idx,:] = np.array([numbers]).astype(float)
    return mat

def getCameraParam(colorOrZ = "color"):
    if(colorOrZ == "color"):
        fx_rgb = 5.1885790117450188e+02
        fy_rgb = 5.1946961112127485e+02
        cx_rgb = 3.2558244941119034e+02
        cy_rgb = 2.5373616633400465e+02
        C = np.array([[fx_rgb, 0, cx_rgb], [0, fy_rgb, cy_rgb], [0, 0, 1]])
    else:
        fx_d = 5.8262448167737955e+02
        fy_d = 5.8269103270988637e+02
        cx_d = 3.1304475870804731e+02
        cy_d = 2.3844389626620386e+02
        C = np.array([[fx_d, 0, cx_d], [0, fy_d, cy_d], [0, 0, 1]])
    return C
def cropCamera(C):
    C[0,2] = C[0,2] - 40
    C[1,2] = C[1,2] - 45
    return C
