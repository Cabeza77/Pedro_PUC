
# coding: utf-8

# In[1]:

import numpy as np
from scipy.optimize import least_squares


# In[2]:

def error(params, points): #funcion a minimizar
    A,B,C = params
    A,B,C = np.array([A,B,C])/np.linalg.norm([A,B,C])
    dist = dist_point_plane(A,B,C,0,points)
    return np.sum(dist)

def dist_point_plane(a,b,c,d,point):
    x,y,z = point[:,0],point[:,1],point[:,2]
    dist = np.abs(a*x+b*y+c*z+d)/np.linalg.norm([a,b,c])
    return dist

def plane_fit(points):
    if len(points) == 3:
        vect_dir1 = points[0] - points[1]
        vect_dir2 = points[0] - points[2]
        vect_normal = np.cross(vect_dir1,vect_dir2)
        A,B,C = vect_normal[0], vect_normal[1], vect_normal[2]
    else:
        x0 = [.5,.5,.5]
        lo = [-1,-1,-1]
        hi = [1,1,1]
        result = least_squares(error, x0, args=(points,),bounds=(lo, hi),jac='3-point') 
        A,B,C = result['x']
    return A,B,C

# In[5]:

def RANSAC_3(set_points, number_iterations, threshold):
# set_points: conjunto de puntos de la forma [[x1,y1,z1]...[xn,yn,zn]]
# number_iterations: numero de iteraciones
# threshold: distancia (euclideana) para considerar si un punto esta acorde con el modelo o no 
    
    iterations = 0
    bestScore = np.inf
    
    while iterations < number_iterations:
        
        permutation = np.random.permutation(len(set_points)) #Se ordenan al azar los indices de los puntos 
        points_select = set_points[permutation[0:3]] #Se seleccionan los 3 primeros puntos random
        
        A,B,C = plane_fit(points_select) #Crea los parametros del modelo con los 'points_select'
        
        ###
        error_fit = np.abs(A*set_points[:,0] + B*set_points[:,1] + C*set_points[:,2]) / np.linalg.norm([A,B,C])
        #Se calculan las distancias de todos los puntos al modelo
        
        points_in = set_points[np.where(error_fit <= threshold)[0]] 
        points_out = set_points[np.where(error_fit > threshold)[0]]
        #Aquellos puntos con distancias menores al 'threshold' se consideran dentro y los que no afuera
        
        Score_under_threshold = np.sum(error_fit[np.where(error_fit <= threshold)[0]])
        Score_over_threshold = len(points_out)*threshold
        #La suma de las distancias es sumada para los puntos que estan dentro y si esta afuera se suma 'threshold'
        
        currentScore = Score_under_threshold + Score_over_threshold
        ###
        
        '''
        #Esto es equivalente a lo anterior encerrado en ### solo que contiene un for()
        currentScore = 0
        points_in, points_out = [],[]
        
        for point in points_unselect:
            error = np.abs(A*point[0] + B*point[1] + C*point[2]) / np.linalg.norm([A,B,C])
            
            if error < threshold:
                currentScore += error
                points_in.append(point)
                
            else:
                currentScore += threshold
                points_out.append(point)
        '''
        
        #La menor distancia al modelo que se encuentra en todas las iteraciones es la que se 
        #Considera la mejor y sus parametros son guardados
        if currentScore < bestScore:
            bestScore = currentScore
            A_best, B_best, C_best = [A, B, C]/np.linalg.norm([A,B,C])
            points_in_best = points_in
            points_out_best = points_out
            
        iterations += 1
        
    return np.array(points_in_best), np.array(points_out_best), [A_best, B_best, C_best], bestScore

# In[6]:

def RANSAC_N(set_points, min_points, number_iterations, threshold, near_points):
# set_points: conjunto de puntos de la forma [[x1,y1,z1]...[xn,yn,zn]]
# min_points: el minimo de puntos con el que realizar el modelo de plano
# number_iterations: numero de iteraciones
# threshold: distancia (euclideana) para considerar si un punto esta acorde con el modelo o no 
# near_points: puntos minimos necesarios para considerarlos en el ajuste
    
    iterations = 0
    bestScore = np.inf
    
    while iterations < number_iterations:
        
        permutation = np.random.permutation(len(set_points)) #Se ordenan al azar los indices de los puntos 
        points_select = set_points[permutation[0:min_points]] #Se seleccionan 'min_points' puntos random
        
        A,B,C = plane_fit(points_select) #Crea los parametros del modelo con los 'points_select'
        
        error_fit = np.abs(A*set_points[:,0] + B*set_points[:,1] + C*set_points[:,2]) / np.linalg.norm([A,B,C])
        #Se calculan las distancias de todos los puntos al modelo
        
        points_in = set_points[np.where(error_fit <= threshold)[0]] 
        points_out = set_points[np.where(error_fit > threshold)[0]]
        #Aquellos puntos con distancias menores al 'threshold' se consideran dentro y los que no afuera
        
        Score_under_threshold = np.sum(error_fit[np.where(error_fit <= threshold)[0]])
        Score_over_threshold = len(points_out)*threshold
        #La suma de las distancias es sumada para los puntos que estan dentro y si esta afuera se suma 'threshold'
        
        currentScore = Score_under_threshold + Score_over_threshold
        
        #La menor distancia al modelo que se encuentra en todas las iteraciones es la que se 
        #Considera la mejor y sus parametros son guardados
        
        if len(points_in) >= near_points:
			#Si hay suficientes puntos definidos por 'near_points' se considera un buen ajuste
			
			A,B,C = plane_fit(points_in)
			#Se crea un nuevo ajuste tomando en consideración a todos los puntos que cumplen con la condición de threshold
			#Luego se calcula el error asociado al ajuste y se conserva el menor error
			error_fit = np.abs(A*set_points[:,0] + B*set_points[:,1] + C*set_points[:,2]) / np.linalg.norm([A,B,C])
        
			points_in = set_points[np.where(error_fit <= threshold)[0]] 
			points_out = set_points[np.where(error_fit > threshold)[0]]
        
			Score_under_threshold = np.sum(error_fit[np.where(error_fit <= threshold)[0]])
			Score_over_threshold = len(points_out)*threshold
        
			currentScore = Score_under_threshold + Score_over_threshold
        
        
			if currentScore < bestScore:
				bestScore = currentScore
				A_best, B_best, C_best = [A, B, C]/np.linalg.norm([A,B,C])
				points_in_best = points_in
				points_out_best = points_out
            
        iterations += 1
        
    return np.array(points_in_best), np.array(points_out_best), [A_best, B_best, C_best], bestScore

a = np.array([[1,1,1],[2,2,2],[3,3,3]])
b = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
