import numpy as np
import kspaceanalysis as ka

'''
ConDen(pos,grid)
denmom2vel(mapden,mapmom,grid)
interp0(field,grid)
LCE(den,grid)

'''

def ConDen(pos,grid):
    '''
    given positon of particles
    return density fields    
    '''
    den=np.histogramdd(pos,bins=(grid,grid,grid))[0]
    den=den*grid**3/np.sum(den)
    return den
    
def denmom2vel(mapden,mapmom,grid):
    '''
    Using mom&den to cal vel
    '''
	mapvel=np.zeros((grid,grid,grid,3))
	mapmom=mapmom.reshape(grid,grid,grid,3)
	mapden=mapden.reshape(grid,grid,grid)
	mapden[np.where(mapden==0)]=1
	mapvel[:,:,:,0]=mapmom[:,:,:,0]/mapden;interp0(mapvel[:,:,:,0],grid)
	mapvel[:,:,:,1]=mapmom[:,:,:,1]/mapden;interp0(mapvel[:,:,:,1],grid)
	mapvel[:,:,:,2]=mapmom[:,:,:,2]/mapden;interp0(mapvel[:,:,:,2],grid)
	return mapvel

def interp0(field,grid): 
    '''
    For some empty grids, using interpolation to calculate velosity
    only suitable to single zero point
    '''
	field=field.reshape(grid,grid,grid)
	coor=np.where(field==0)
	#print(coor)
	for x in range(len(coor[0])):
		i=coor[0][x]
		j=coor[1][x]
		k=coor[2][x]
		n0=len(np.where(field[i-1:i+2,j-1:j+2,k-1:k+2]==0)[0])
		if n0>0:
			#print(i,j,k,n0)
			field[i,j,k]=np.sum(field[i-1:i+2,j-1:j+2,k-1:k+2])*9.0/8
			
			
def LCE(den,grid):
	'''
	Linear continuity equation
	given density, return velosity
	unscaled
	'''
	den=den.reshape(grid,grid,grid)
	kmod,kx,ky,kz=ka.karray(grid,True)
	denk=np.fft.rfftn(denhalo)
	velxk=np.zeros(denk.shape,dtype=type(denk))
	velyk=np.zeros(denk.shape,dtype=type(denk))
	velzk=np.zeros(denk.shape,dtype=type(denk))
	velxk[1:,1:,1:]=kx[1:,1:,1:]*denk[1:,1:,1:,]/kmod[1:,1:,1:,]**2*(1j)
	velyk[1:,1:,1:]=ky[1:,1:,1:]*denk[1:,1:,1:,]/kmod[1:,1:,1:,]**2*(1j)
	velzk[1:,1:,1:]=kz[1:,1:,1:]*denk[1:,1:,1:,]/kmod[1:,1:,1:,]**2*(1j)

	vel=np.zeros(grid,grid,grid,3)
	vel[:,:,:,0]=np.fft.irfftn(velxk)
	vel[:,:,:,1]=np.fft.irfftn(velyk)
	vel[:,:,:,2]=np.fft.irfftn(velzk)
	
	return vel
	
	
	
