import numpy as np
import numba
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

precise=0.1 #for bins of k, same for whole file

‘’‘
Functions in this files:
    karray(grid, BE=False)
    karray2d(grid, BE=False)
    
    CalPS(field,grid,mapscale=1,kscale=1)
    CalPS2d(field,grid,mapscale=1,kscale=1)
    CalPSvector(field,grid,mapscale=1,kscale=1)
    CalPSvector2d(field,grid,mapscale=1,kscale=1)
    
    CalR(field1, field2, grid,kscale=1)
    CalR2d(field1, field2, grid, kscale=1)
    CalRvector(field1, field2, grid, kscale=1)
    CalRvector2d(field1, field2, grid, kscale=1)
    
    BEdecomposition(field,grid)
    
    def ThetaPDF(map1k,map2k,kmod,kvalue,kscale=1)
    

’‘’

def karray(grid, BE=False):
	'''
    Return the mod of k in the fft order,3d
	If BE is true, return kx,ky,kz
	'''
	kmod =np.zeros((grid,grid,grid//2+1))
	if BE:
		kx =np.zeros((grid,grid,grid//2+1))
		ky =np.zeros((grid,grid,grid//2+1))
		kz =np.zeros((grid,grid,grid//2+1))
	for ii in [0,1]:
		for jj in [0,1]:
			kmod[ii*(grid//2)+1:(ii+1)*(grid//2+1)-ii,jj*(grid//2)+1:(jj+1)*(grid//2+1)-jj,1:]=np.sqrt(np.arange(1-ii*(grid//2),grid//2+1-ii*(1+grid//2),1).reshape(-1,1,1)**2+np.arange(1-jj*(grid//2),grid//2-jj*(1+grid//2)+1,1).reshape(1,-1,1)**2+np.arange(1,grid//2+1,1).reshape(1,1,-1)**2)
			if BE:				
				kx[ii*(grid//2)+1:(ii+1)*(grid//2+1)-ii,jj*(grid//2)+1:(jj+1)*(grid//2+1)-jj,1:]=np.arange(1-ii*(grid//2),grid//2+1-ii*(1+grid//2),1).reshape(-1,1,1)
				ky[ii*(grid//2)+1:(ii+1)*(grid//2+1)-ii,jj*(grid//2)+1:(jj+1)*(grid//2+1)-jj,1:]=np.arange(1-jj*(grid//2),grid//2-jj*(1+grid//2)+1,1).reshape(1,-1,1)
				kz[ii*(grid//2)+1:(ii+1)*(grid//2+1)-ii,jj*(grid//2)+1:(jj+1)*(grid//2+1)-jj,1:]=np.arange(1,grid//2+1,1).reshape(1,1,-1)
	if BE: return kmod, kx,ky,kz
	return kmod

def karray2d(grid, BE=False):
	'''
	Return the mod of k in the fft order, 2d
	If BE is true, return kx,ky,kz
	'''
	kmod =np.zeros((grid,grid//2+1))
	if BE:
		kx =np.zeros((grid,grid//2+1))
		ky =np.zeros((grid,grid//2+1))
		kz =np.zeros((grid,grid//2+1))
	for ii in [0,1]:

			kmod[ii*(grid//2)+1:(ii+1)*(grid//2+1)-ii,1:]=np.sqrt(np.arange(1-ii*(grid//2),grid//2+1-ii*(1+grid//2),1).reshape(-1,1)**2+np.arange(1,grid//2+1,1).reshape(1,-1)**2)
			if BE:				
				kx[ii*(grid//2)+1:(ii+1)*(grid//2+1)-ii,1:]=np.arange(1-ii*(grid//2),grid//2+1-ii*(1+grid//2),1).reshape(-1,1)
				ky[ii*(grid//2)+1:(ii+1)*(grid//2+1)-ii,1:]=np.arange(1,grid//2+1,1).reshape(1,-1)
	if BE: return kmod, kx,ky
	return kmod

def CalPS(field,grid,mapscale=1,kscale=1):
    '''
    Calculate a power spectrum of a 3-D scalar field
    mapscale=(boxlen/grid**2)*3
    kscale=2*np.pi/grid    
    '''
	field=field.reshape(grid,grid,grid)
	fieldk=np.fft.rfftn(field)
	kmod=karray(grid)
	kmodcount=(np.log10(kmod[1:,1:,1:])/precise).astype(np.int32)
	count=np.bincount(kmodcount.flatten())
	for x1 in range(len(count)):
		if np.min(count[x1:])>0:break
	ps=np.bincount(kmodcount.flatten(),weights=np.abs(fieldk[1:,1:,1:].flatten())**2)
	k=np.bincount(kmodcount.flatten(),weights=kmod[1:,1:,1:].flatten())
	k=kscale*k[x1:]/count[x1:]
	ps=mapscale*ps[x1:]/count[x1:]
	
	return ps,k

def CalPS2d(field,grid,mapscale=1,kscale=1):
    '''
    Calculate a power spectrum of a 2-D scalar field
    mapscale=(boxlen/grid**2)*3
    kscale=2*np.pi/grid    
    '''
	field=field.reshape(grid,grid)
	fieldk=np.fft.rfft2(field)
	kmod=karray2d(grid)
	kmodcount=(np.log10(kmod[1:,1:])/precise).astype(np.int32)
	count=np.bincount(kmodcount.flatten())
	for x1 in range(len(count)):
		if np.min(count[x1:])>0:break
	ps=np.bincount(kmodcount.flatten(),weights=np.abs(fieldk[1:,1:].flatten())**2)
	k=np.bincount(kmodcount.flatten(),weights=kmod[1:,1:].flatten())
	k=kscale*k[x1:]/count[x1:]
	ps=mapscale*ps[x1:]/count[x1:]

	return ps,k

def CalPSvector(field,grid,mapscale=1,kscale=1):
    '''
    Calculate a power spectrum of a 3-D vector field (grid,grid,grid,3)
    mapscale=(boxlen/grid**2)*3
    kscale=2*np.pi/grid    
    '''
	field=field.reshape(grid,grid,grid,3)
	fieldxk=np.fft.rfftn(field[:,:,:,0])
	fieldyk=np.fft.rfftn(field[:,:,:,1])
	fieldzk=np.fft.rfftn(field[:,:,:,2])
	kmod=karray(grid)
	kmodcount=(np.log10(kmod[1:,1:,1:])/precise).astype(np.int32)
	count=np.bincount(kmodcount.flatten())
	for x1 in range(len(count)):
		if np.min(count[x1:])>0:break
	ps=np.bincount(kmodcount.flatten(),weights=(np.abs(fieldxk[1:,1:,1:])**2+np.abs(fieldyk[1:,1:,1:])**2+np.abs(fieldzk[1:,1:,1:])**2).flatten())
	k=np.bincount(kmodcount.flatten(),weights=kmod[1:,1:,1:].flatten())
	k=kscale*k[x1:]/count[x1:]
	ps=mapscale*ps[x1:]/count[x1:]
	
	return ps,k

def CalPSvector2d(field,grid,mapscale=1,kscale=1):
    '''
    Calculate a power spectrum of a 3-D vector field (grid,grid,2)
    mapscale=(boxlen/grid**2)*3
    kscale=2*np.pi/grid    
    '''
	field=field.reshape(grid,grid,2)
	fieldxk=np.fft.rfftn(field[:,:,0])
	fieldyk=np.fft.rfftn(field[:,:,1])
	
	kmod=karray(grid)
	kmodcount=(np.log10(kmod[1:,1:])/precise).astype(np.int32)
	count=np.bincount(kmodcount.flatten())
	for x1 in range(len(count)):
		if np.min(count[x1:])>0:break
	ps=np.bincount(kmodcount.flatten(),weights=(np.abs(fieldxk[1:,1:])**2+np.abs(fieldyk[1:,1:])**2).flatten())
	k=np.bincount(kmodcount.flatten(),weights=kmod[1:,1:].flatten())
	k=kscale*k[x1:]/count[x1:]
	ps=mapscale*ps[x1:]/count[x1:]
	
	return ps,k

def CalR(field1, field2, grid,kscale=1):
    '''
    Calculate the correlation parameter of 2 scalar 3D fields 
    kscale=2*np.pi/grid
    '''
	field1=field1.reshape(grid,grid,grid)
	field1k=np.fft.rfftn(field1)
	field2=field2.reshape(grid,grid,grid)
	field2k=np.fft.rfftn(field2)
	kmod=karrayfor(grid)
	kmodcount=(np.log10(kmod[1:,1:,1:])/precise).astype(np.int32)
	count=np.bincount(kmodcount.flatten())
	for x1 in range(len(count)):
		if np.min(count[x1:])>0:break
	ps1=np.bincount(kmodcount.flatten(),weights=np.abs(field1k[1:,1:,1:].flatten())**2)
	ps1=ps1[x1:]/count[x1:]
	ps2=np.bincount(kmodcount.flatten(),weights=np.abs(field2k[1:,1:,1:].flatten())**2)
	ps2=ps2[x1:]/count[x1:]
	ps12=np.bincount(kmodcount.flatten(),weights=np.real(field1k[1:,1:,1:]*np.conjugate(field2k[1:,1:,1:])).flatten())
	ps12=ps12[x1:]/count[x1:]
	k=np.bincount(kmodcount.flatten(),weights=kmod[1:,1:,1:].flatten())
	k=k[x1:]/count[x1:]
	return ps12/np.sqrt(ps1*ps2),k*kscale

def CalR2d(field1, field2, grid, kscale=1):
    '''
    Calculate the correlation parameter of 2 scalar 2D fields 
    kscale=2*np.pi/grid
    '''
	field1=field1.reshape(grid,grid)
	field1k=np.fft.rfft2(field1)
	field2=field2.reshape(grid,grid)
	field2k=np.fft.rfft2(field2)
	kmod=karrayfor2d(grid)
	kmodcount=(np.log10(kmod[1:,1:])/precise).astype(np.int32)
	count=np.bincount(kmodcount.flatten())
	for x1 in range(len(count)):
		if np.min(count[x1:])>0:break
	ps1=np.bincount(kmodcount.flatten(),weights=np.abs(field1k[1:,1:].flatten())**2)
	ps1=ps1[x1:]/count[x1:]

	ps2=np.bincount(kmodcount.flatten(),weights=np.abs(field2k[1:,1:].flatten())**2)
	ps2=ps2[x1:]/count[x1:]
	ps12=np.bincount(kmodcount.flatten(),weights=np.real(field1k[1:,1:]*np.conjugate(field2k[1:,1:])).flatten())
	ps12=ps12[x1:]/count[x1:]
	k=np.bincount(kmodcount.flatten(),weights=kmod[1:,1:].flatten())
	k=k[x1:]/count[x1:]
	return ps12/np.sqrt(ps1*ps2),k*kscale

def CalRvector(field1, field2, grid, kscale=1):
    '''
    Calculate the correlation parameter of 2 vector 3Dfields 
    kscale=2*np.pi/grid
    '''
	field1=field1.reshape(grid,grid,grid,3)
	field2=field2.reshape(grid,grid,grid,3)
	map1xk=np.fft.rfftn(field1[:,:,:,0])
	map1yk=np.fft.rfftn(field1[:,:,:,1])
	map1zk=np.fft.rfftn(field1[:,:,:,2])
	map2xk=np.fft.rfftn(field2[:,:,:,0])
	map2yk=np.fft.rfftn(field2[:,:,:,1])
	map2zk=np.fft.rfftn(field2[:,:,:,2])
	kmod=karrayfor(grid)
	
	kmodcount=(np.log10(kmod[1:,1:,1:])/precise).astype(np.int32)
	count=np.bincount(kmodcount.flatten())
	for x1 in range(len(count)):
		if np.min(count[x1])>0:break
	kbin=np.bincount(kmodcount.flatten(),weights=kmod[1:,1:,1:].flatten())
	kbin=kbin[x1:]/count[x1:]
	ps1=np.bincount(kmodcount.flatten(),weights=(np.abs(map1xk[1:,1:,1:])**2+np.abs(map1yk[1:,1:,1:])**2+np.abs(map1zk[1:,1:,1:])**2).flatten());ps1=ps1[x1:]/count[x1:];
	ps2=np.bincount(kmodcount.flatten(),weights=(np.abs(map2xk[1:,1:,1:])**2+np.abs(map2yk[1:,1:,1:])**2+np.abs(map2zk[1:,1:,1:])**2).flatten());ps2=ps2[x1:]/count[x1:];
	ps12=np.bincount(kmodcount.flatten(),weights=np.real(map1xk[1:,1:,1:]*np.conjugate(map2xk[1:,1:,1:])+map1yk[1:,1:,1:]*np.conjugate(map2yk[1:,1:,1:])+map1zk[1:,1:,1:]*np.conjugate(map2zk[1:,1:,1:])).flatten());ps12=ps12[x1:]/count[x1:]
	return ps12/np.sqrt(ps1*ps2),kbin*kscale

def CalRvector2d(field1, field2, grid, kscale=1):
    '''
    Calculate the correlation parameter of 2 vector 2D fields 
    kscale=2*np.pi/grid
    '''
	field1=field1.reshape(grid,grid,2)
	field2=field2.reshape(grid,grid,2)
	map1xk=np.fft.rfftn(field1[:,:,0])
	map1yk=np.fft.rfftn(field1[:,:,1])
	map2xk=np.fft.rfftn(field2[:,:,0])
	map2yk=np.fft.rfftn(field2[:,:,1])
	
	kmod=karray2d(grid)
	kmodcount=(np.log10(kmod[1:,1:])/precise).astype(np.int32)
	count=np.bincount(kmodcount.flatten())
	for x1 in range(len(count)):
		if np.min(count[x1])>0:break
	kbin=np.bincount(kmodcount.flatten(),weights=kmod[1:,1:].flatten())
	kbin=kbin[x1:]/count[x1:]
	ps1=np.bincount(kmodcount.flatten(),weights=(np.abs(map1xk[1:,1:])**2+np.abs(map1yk[1:,1:])**2).flatten())
    ps1=ps1[x1:]/count[x1:];
	ps2=np.bincount(kmodcount.flatten(),weights=(np.abs(map2xk[1:,1:])**2+np.abs(map2yk[1:,1:])**2).flatten())
    ps2=ps2[x1:]/count[x1:];
	ps12=np.bincount(kmodcount.flatten(),weights=np.real(map1xk[1:,1:]*np.conjugate(map2xk[1:,1:])+map1yk[1:,1:]*np.conjugate(map2yk[1:,1:])+map1zk[1:,1:]*np.conjugate(map2zk[1:,1:])).flatten())
    ps12=ps12[x1:]/count[x1:]
	return ps12/np.sqrt(ps1*ps2),kbin*kscale

def BEdecomposition(field,grid):
    '''
    Get B, E modes of a 3-d vector field
    '''
    field.reshape(grid,grid,grid,3)
    xk=np.fft.rfftn(field1[:,:,:,0])
	yk=np.fft.rfftn(field1[:,:,:,1])
	zk=np.fft.rfftn(field1[:,:,:,2])
    
    kmod,kx,ky,kz=karray(grid,True)
    
	Ek=np.zeros(kmod.shape,dtype=complex)
	Ek[1:,1:,1:]=(kx*xk+ky*yk+kz*zk)[1:,1:,1:]/kmod[1:,1:,1:]
	Bxk=np.zeros(kmod.shape,dtype=complex)
	Byk=np.zeros(kmod.shape,dtype=complex)
	Bzk=np.zeros(kmod.shape,dtype=complex)
	Bxk[1:,1:,1:]=xk[1:,1:,1:]-(Ek*kx)[1:,1:,1:]/kmod[1:,1:,1:]
	Byk[1:,1:,1:]=yk[1:,1:,1:]-(Ek*ky)[1:,1:,1:]/kmod[1:,1:,1:]
	Bzk[1:,1:,1:]=zk[1:,1:,1:]-(Ek*kz)[1:,1:,1:]/kmod[1:,1:,1:]
    
    E=np.zeros(grid,grid,grid,3)
    B=np.zeros(grid,grid,grid,3)
    
    temp=np.fft.irfftn(Ek)
    E[:,:,:,0]=kx/kmod*temp
    E[:,:,:,1]=ky/kmod*temp
    E[:,:,:,2]=kz/kmod*temp
    
    B[:,:,:,0]=np.fft.irfftn(Bxk)
    B[:,:,:,1]=np.fft.irfftn(Byk)
    B[:,:,:,2]=np.fft.irfftn(Bzk)
	
	return E,B

def ThetaPDF(map1k,map2k,kmod,kvalue,kscale=1):
    '''
    PDF for delta theta of 2 field at kvalue
    '''
	map1k=map1k[1:,1:,1:].flatten();map2k=map2k[1:,1:,1:].flatten();kmodcount=kscale*kmod[1:,1:,1:].flatten()
	thetaall=np.angle(map1k/map2k)
	for k in kvalue:
		theta=thetaall[np.where(abs(kmodcount-k)<0.02)]
		print("k=",k)
		print("Number of theta:",len(theta))
		print("Max:",np.max(theta),"Min:",np.min(theta), "Mean:", np.mean(theta))	
		n,tx=np.histogram(theta,30,range=[-3.14,3.14])	
		print(n)
		print(tx)
		np.save("PDFtheta/k_"+str(k), theta)

