import matplotlib.pyplot as plt
import numpy as np

# payload fairing (spherically blunted tangent ogive)
def calc_payload_fairing(diameter):	
	R = 0.5 * diameter				# radius
	Lo= 1.225 * diameter			# Length of ogive part
	rho= ( R**2 + Lo**2 ) / (2*R)	# rho = (R^2 + L^2)/2R
	rn = 0.28 * diameter			# nose radius

	# output values
	x_out = []; y_out = []; yne_out = []

	# tangency point (circle meets ogive)
	xo = Lo - np.sqrt((rho - rn)**2 - (rho - R)**2)
	yt = (rn * (rho - R) ) / (rho - rn)
	xt = xo - np.sqrt(rn**2 - yt**2)
	xa = xo - rn
	
	# draw blunt sphere
	x = np.linspace(xa, xt)
	y = np.sqrt(rn**2 - (xo-x)**2)
	yne  = [-t for t in y]	
	# collect array values
	x_out = np.append(x_out, x)
	y_out = np.append(y_out, y)
	yne_out=np.append(yne_out, yne)
	
	# ogive from tangency
	x = np.linspace(xt, Lo)
	y = np.sqrt(rho**2 - (Lo - x)**2) + R - rho
	yne  = [-t for t in y]	
	# collect array values
	x_out = np.append(x_out, x)
	y_out = np.append(y_out, y)
	yne_out=np.append(yne_out, yne)
	
	# cylinder part
	x_i = x_out[-1]; y_i = y_out[-1];	
	Lc = Lo + (0.75 * diameter)		# Length of cylinder part
	x = np.linspace(x_i, Lc)
	y = x/x * y_i
	yne  = [-t for t in y]
	# collect array values
	x_out = np.append(x_out, x)
	y_out = np.append(y_out, y)
	yne_out=np.append(yne_out, yne)		
	
	# frustum part
	x_i = x_out[-1]; y_i = y_out[-1];
	Lf = Lc + (0.275 * diameter)	# Length of frustum part
	x_end = Lf; y_end = 0.8 * diameter/2;
	# equation of the slanting line
	slope = (y_end - y_i) / (x_end - x_i)
	intercept = ( (x_end * y_i) - (x_i * y_end) ) / ( x_end - x_i)
	# conical part
	x = np.linspace(x_i, Lf)
	y = x * slope + intercept
	yne  = [-t for t in y]
	# collect array values
	x_out = np.append(x_out, x)
	y_out = np.append(y_out, y)
	yne_out=np.append(yne_out, yne)	
	
	# other dimensions
	a_dim = (R, Lo, rn, rho, xo, xa, xt, yt)
	
	return a_dim, x_out, y_out, yne_out

	
# find the nearest index in the list for the given value
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]	

# 2D plot
def plot2D(ax, d, x, y, yne, ifdim=False):
	# unpack the dimension values
	R, Lo, rn, rho, xo, xa, xt, yt = d

	# set correct aspect ratio
	ax.set_aspect('equal')
	
	# plot	
	ax.plot(x, y, color='g'); ax.plot(x, yne, color='g');

	if ifdim:
		# apex length
		text = ' xa = '+ str(round( xa,1 ) )
		ax.plot(0, 0, '.' ); ax.plot(xa, 0, '+' );
		ax.annotate( "", [0, 0], [xa, 0] , arrowprops=dict(lw=0.5, arrowstyle='<-') )	
		ax.text(0.1, -0.1, text, fontsize=9 )
				
		# center of the spherical nose cap
		text = ' xo = '+ str(round( xo,1 ) ) 
		ax.plot(0, -0.2, '.' ); ax.plot(xo, -0.2, '+' );
		ax.annotate( "", [0, -0.2], [xo, -0.2] , arrowprops=dict(lw=0.5, arrowstyle='<-') )	
		ax.text(xo/2-0.5, -0.5, text, fontsize=9 )
		
		# tangency point
		text = ' [xt, yt] = {'+ str(round(xt,1)) +','+ str(round(-yt,1)) + '} '
		ax.plot(xt, yt, '.' ); ax.plot(xt, -yt, '.' )
		ax.text(-xt+2, -yt-0.5, text, fontsize=9 )
				
		# blunt sphere radius, inlet radial line [xo, 0] to [xt, yt]
		text = ' Rn = '+ str(round(rn,1)) + ' \n [0.28 * D] '
		ax.plot(xo, 0, '.' ); ax.plot(xt, yt, '+' )
		# draw dimension from [xo, 0] to [xt, yt]
		ax.annotate( "", [xo, 0], [xt, yt] , arrowprops=dict(lw=0.5, arrowstyle='<-') )
		ax.text((xo+xt)/2, (0+yt)/2, text, fontsize=9 )
				
		# ogive radius 
		text = ' rho = '+ str(round(rho,1)) + ' \n [($R^2$ + $L^2$)/2R] ' #$10^1$
		# array mid point value
		x_near = Lo/2
		x_index, x_middle = find_nearest(x, x_near)		
		y_middle = y[x_index]
		ax.plot(Lo, rho-R, '+' ); ax.plot(x_middle, -y_middle, '+' )
		ax.annotate( "", [Lo, rho-R], [x_middle, -y_middle] , arrowprops=dict(lw=0.5, arrowstyle='<-') )	
		ax.text((Lo+x_middle)/2-0.8, ((rho-R)-y_middle)/2+0.5, text, fontsize=9 )
		
		# diameter
		text = ' R = '+ str(round(R,1)) + ' '
		ax.plot(Lo, 0, '.' ); ax.plot(Lo, R, '+' );
		ax.annotate( "", [Lo, 0], [Lo, R] , arrowprops=dict(lw=0.5, arrowstyle='<-') )	
		ax.text(Lo, R/2, text, fontsize=9 )
				
		# ogive length
		text = ' Lo = '+ str(round( Lo,1 ) ) + ' \n [1.225 * D] '
		ax.plot(0, -0.4, '.' ); ax.plot(Lo, -0.4, '+' );
		ax.annotate( "", [0, -0.4], [Lo, -0.4] , arrowprops=dict(lw=0.5, arrowstyle='<-') )	
		ax.text((Lo)/2, -0.7, text, fontsize=9 )
		
		# cylinder length
		Lc = Lo + (0.75 * 2 * R)	
		text = ' Lc = '+ str(round( (0.75 * 2 * R),1 ) ) + ' \n [0.75 * D] '
		ax.plot(Lo, -0.3, '.' ); ax.plot(Lc, -0.3, '+' );
		ax.annotate( "", [Lo, -0.3], [Lc, -0.3] , arrowprops=dict(lw=0.5, arrowstyle='<-') )	
		ax.text((Lo+Lc)/2, -0.6, text, fontsize=9 )
		
		# frustum length
		Lf = Lc + (0.275 *  2 * R)
		text = ' Lf = '+ str(round( (0.275 * 2 * R),1 ) ) + ' \n [0.275 * D] '
		ax.plot(Lc, -0.2, '.' ); ax.plot(Lf, -0.2, '+' );
		ax.annotate( "", [Lc, -0.2], [Lf, -0.2] , arrowprops=dict(lw=0.5, arrowstyle='<-') )
		ax.text((Lc+Lf)/2, -0.5, text, fontsize=9 )
		
		# end diameter
		Rt = 0.8 * R
		text = ' Re = '+ str(round(Rt,1)) + ' '
		Lt = 2.25 * 2 * R
		ax.plot(Lt, 0, '.' ); ax.plot(Lt, Rt, '+' );
		ax.annotate( "", [Lt, 0], [Lt, Rt] , arrowprops=dict(lw=0.5, arrowstyle='<-') )	
		ax.text(Lt, Rt/2, text, fontsize=9 )		

	# axis
	ax.axhline(color='black', lw=0.5, linestyle="dashed")
	ax.axvline(color='black', lw=0.5, linestyle="dashed")		
	
	# grids
	ax.grid()
	ax.minorticks_on()
	ax.grid(which='major', linestyle='-', linewidth='0.5') # , color='red'
	ax.grid(which='minor', linestyle=':', linewidth='0.5') # , color='black'	

	return

# ring of radius r, height h, base point a
def ring(r, h, a=0, n_theta=30, n_height=10):
    theta = np.linspace(0, 2*np.pi, n_theta)
    v = np.linspace(a, a+h, n_height )
    theta, v = np.meshgrid(theta, v)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = v
    return x, y, z

# Set 3D plot axes to equal scale. 
# Required since `ax.axis('equal')` and `ax.set_aspect('equal')` don't work on 3D.
def set_axes_equal_3d(ax: plt.Axes):
    """	
    https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)
    return

# set axis limits
def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])
    return

# 3d plot
def plot3D(ax, d, x, y, yne):
	# unpack the dimension values
	R, Lo, rn, rho, xo, xa, xt, yt = d
	
	# ring thickness
	thick = 5 * (x[1] - x[0]) # 0.01
	
	# draw multiple rings to create 3d structure
	for i in range(len(y)):
		X, Y, Z = ring(y[i], thick, x[i])
		ax.plot_surface(X, Y, Z, color='g')	
		
	# set correct aspect ratio
	ax.set_box_aspect([1,1,1])
	set_axes_equal_3d(ax)
	# set view
	ax.view_init(-170, -15)
	return

def plot(d, x, y, yne):
	# set flag if dimensions in 2d plot are needed
	ifdim = True
		
	# Plot 3d view
	fig = plt.figure(figsize=(12,9))
	# plot some 2d information
	ax1 = fig.add_subplot(121)
	plot2D(ax1, d, x, y, yne, ifdim)
	# plot 3d view
	ax2 = fig.add_subplot(122, projection='3d')
	plot3D(ax2, d, x, y, yne)	
	# show
	fig.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.show()
	return
	
if __name__ == '__main__':
	# fairing diameter
	fairing_dia  = 5.0  
	
	# calculate values
	d, x, y, yne = calc_payload_fairing(fairing_dia)
	
	# and plot it
	plot(d, x, y, yne)


