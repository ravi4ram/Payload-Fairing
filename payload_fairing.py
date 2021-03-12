import matplotlib.pyplot as plt
import numpy as np

# payload fairing (spherically blunted cone)
def calc_pslv_fairing(diameter):	
	# output values
	x_out = []; y_out = []; yne_out = []
	
	"""
	sphere radius, rs = 1.176
	tangency point, xt = rn * sin(theta), theta - half cone angle - 15 degrees
	cone length, Lo = 1.866 * D
	Lc = 1.5 * fairing_dia;     // cylinder length
	Lf = 0.153 * fairing_dia;   // frustum length
	Re = 0.7 * fairing_dia/2;   // frustum end radius
	"""
	R = 0.5 * diameter		# fairing radius
	rn = 0.294 * diameter	# nose radius
	theta_nose = np.radians(15)
	theta_boat = np.radians(20)

	# R/Lo = tan(theta_nose)
	Lo = R / np.tan(theta_nose)
		
	# tangency point (circle meets cone)
	xt = (Lo**2 / R) * np.sqrt( rn**2 / (R**2 + Lo**2) )
	yt = xt * (R / Lo)		
	xo = xt + np.sqrt( rn**2 - yt**2)
	xa = xo - rn

	# draw blunt sphere
	x = np.linspace(xa, xt)
	# to avoid closer to zero issues on y = np.sqrt(rn**2 - (xo-x)**2)
	b = (rn**2 - (xo-x)**2)
	# set numbers that are less than tol to zero
	b[np.abs(b) < 1e-10] = 0
	y = np.sqrt(b)
	yne  = [-t for t in y]	
	# collect array values
	x_out = np.append(x_out, x)
	y_out = np.append(y_out, y)
	yne_out=np.append(yne_out, yne)

	# cone from tangency
	x = np.linspace(xt, Lo)
	y = x * (R / Lo)
	yne  = [-t for t in y]	
	# collect array values
	x_out = np.append(x_out, x)
	y_out = np.append(y_out, y)
	yne_out=np.append(yne_out, yne)

	# cylinder part
	x_i = x_out[-1]; y_i = y_out[-1];	
	Lc = Lo + (1.5 * diameter)		# Length of cylinder part
	x = np.linspace(x_i, Lc)
	y = x/x * y_i
	yne  = [-t for t in y]
	# collect array values
	x_out = np.append(x_out, x)
	y_out = np.append(y_out, y)
	yne_out=np.append(yne_out, yne)	
	
	# frustum part
	x_i = x_out[-1]; y_i = y_out[-1];
	Lf = Lc + (0.153 * diameter)	# Length of frustum part
	x_end = Lf; y_end = 0.7 * diameter/2;
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
	rho = 0; # for gslv compatibility
	a_dim = (R, Lo, rn, rho, xo, xa, xt, yt)
	
	return a_dim, x_out, y_out, yne_out


# payload fairing (spherically blunted tangent ogive)
def calc_gslv_fairing(diameter):	

	# output values
	x_out = []; y_out = []; yne_out = []

	"""
	sphere radius, rs = 1.330
	tangency point, xt = 0.6615
	ogive length, Lo = 1.225 * D
	frustum length,  Lf = 0.2576 * D = 1.288
	aft frustum dia, xf = 0.8208 * D = 4.104
	cylinder length, Lc = 0.75 * D = 3.750	
	"""
	R = 0.5 * diameter		# fairing radius
	rho= 10					# ogive radius
	rn = 0.266 * diameter	# nose radius

	# rho = ( R**2 + Lo**2 ) / (2*R)
	Lo = np.sqrt( (rho * diameter) - R**2)

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
	Lf = Lc + (0.2576 * diameter)	# Length of frustum part
	x_end = Lf; y_end = 0.8208 * diameter/2;
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


# 2D pslv plot
def plot_pslv_2D(ax, d, x, y, yne, ifdim=False):
	# unpack the dimension values
	R, Lo, rn, rho, xo, xa, xt, yt = d

	# set correct aspect ratio
	ax.set_aspect('equal')
	
	# plot	
	ax.plot(x, y, color='g'); ax.plot(x, yne, color='g');

	if ifdim:
		# apex length
		text = ' xa = '+ str(round( xa,1 ) )
		arrow(ax, [0, 0], [xa, 0], text, [0.1, -0.1])
				
		# center of the spherical nose cap
		text = ' xo = '+ str(round( xo,1 ) ) 
		arrow(ax, [0, -0.2], [xo, -0.2], text, [xo/2-0.5, -0.5])
		
		# tangency point
		text = ' [xt, yt] = {'+ str(round(xt,1)) +','+ str(round(-yt,1)) + '} '
		ax.plot(xt, yt, '.' ); ax.plot(xt, -yt, '.' )
		ax.text(xt, -yt-0.5, text, fontsize=9 )
				
		# blunt sphere radius, inlet radial line [xo, 0] to [xt, yt]
		text = ' Rn = '+ str(round(rn,1))
		arrow(ax, [xo, 0], [xt, yt], text, [(xo+xt)/2, (0+yt)/2])
		
		# diameter
		text = ' R = '+ str(round(R,1))
		arrow(ax, [Lo, 0], [Lo, R], text, [Lo, R/2])
				
		# ogive length
		text = ' Lo = '+ str(round( Lo,1 ) ) 
		arrow(ax, [0, -0.4], [Lo, -0.4] , text, [(Lo)/2, -0.7])
		
		# cylinder length
		Lc = Lo + (1.5 * 2 * R)	
		text = ' Lc = '+ str(round( (1.5 * 2 * R),1 ) ) 
		arrow(ax, [Lo, -0.3], [Lc, -0.3] , text, [(Lo+Lc)/2, -0.6])
		
		# frustum length
		Lf = Lc + (0.153 *  2 * R)
		text = ' Lf = '+ str(round( (0.153 * 2 * R),1 ) ) 
		arrow(ax, [Lc, -0.2], [Lf, -0.2], text, [(Lc+Lf)/2, -0.5])
		
		# end diameter
		Rt = 0.7 * R
		text = ' Re = '+ str(round(Rt,1))	
		arrow(ax, [Lf, 0], [Lf, Rt], text, [Lf, Rt/2])	

	# axis
	ax.axhline(color='black', lw=0.5, linestyle="dashed")
	ax.axvline(color='black', lw=0.5, linestyle="dashed")		
	
	# grids
	ax.grid()
	ax.minorticks_on()
	ax.grid(which='major', linestyle='-', linewidth='0.5') # , color='red'
	ax.grid(which='minor', linestyle=':', linewidth='0.5') # , color='black'	

	return


# 2D gslv plot
def plot_gslv_2D(ax, d, x, y, yne, ifdim=False):
	# unpack the dimension values
	R, Lo, rn, rho, xo, xa, xt, yt = d

	# set correct aspect ratio
	ax.set_aspect('equal')
	
	# plot	
	ax.plot(x, y, color='g'); ax.plot(x, yne, color='g');

	if ifdim:
		# apex length
		text = ' xa = '+ str(round( xa,1 ) )
		arrow(ax, [0, 0], [xa, 0], text, [0.1, -0.1])
				
		# center of the spherical nose cap
		text = ' xo = '+ str(round( xo,1 ) ) 
		arrow(ax, [0, -0.2], [xo, -0.2], text, [xo/2-0.5, -0.5])
		
		# tangency point
		text = ' [xt, yt] = {'+ str(round(xt,1)) +','+ str(round(-yt,1)) + '} '
		ax.plot(xt, yt, '.' ); ax.plot(xt, -yt, '.' )
		ax.text(-xt+2, -yt-0.5, text, fontsize=9 )
				
		# blunt sphere radius, inlet radial line [xo, 0] to [xt, yt]
		text = ' Rn = '+ str(round(rn,1)) 
		arrow(ax, [xo, 0], [xt, yt], text, [(xo+xt)/2, (0+yt)/2])
				
		# ogive radius 
		text = ' rho = '+ str(round(rho,1)) 
		# array mid point value
		x_index, x_middle = find_nearest(x, Lo/2)		
		y_middle = y[x_index]
		arrow(ax, [Lo, rho-R], [x_middle, -y_middle], text, [(Lo+x_middle)/2-0.8, ((rho-R)-y_middle)/2+0.5])
		
		# diameter
		text = ' R = '+ str(round(R,1))
		arrow(ax, [Lo, 0], [Lo, R], text, [Lo, R/2])
				
		# ogive length
		text = ' Lo = '+ str(round( Lo,1 ) ) 
		arrow(ax, [0, -0.4], [Lo, -0.4] , text, [(Lo)/2, -0.7])
		
		# cylinder length
		Lc = Lo + (0.75 * 2 * R)	
		text = ' Lc = '+ str(round( (0.75 * 2 * R),1 ) ) 
		arrow(ax, [Lo, -0.3], [Lc, -0.3] , text, [(Lo+Lc)/2, -0.6])
		
		# frustum length
		Lf = Lc + (0.2576 *  2 * R)
		text = ' Lf = '+ str(round( (0.2576 * 2 * R),1 ) ) 
		arrow(ax, [Lc, -0.2], [Lf, -0.2], text, [(Lc+Lf)/2, -0.5])
		
		# end diameter
		Rt = 0.8208 * R
		text = ' Re = '+ str(round(Rt,1))	
		arrow(ax, [Lf, 0], [Lf, Rt], text, [Lf, Rt/2])	

	# axis
	ax.axhline(color='black', lw=0.5, linestyle="dashed")
	ax.axvline(color='black', lw=0.5, linestyle="dashed")		
	
	# grids
	ax.grid()
	ax.minorticks_on()
	ax.grid(which='major', linestyle='-', linewidth='0.5') # , color='red'
	ax.grid(which='minor', linestyle=':', linewidth='0.5') # , color='black'	

	return


# find the nearest index in the list for the given value
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]	
    
# ar_from = [0, -0.2]; ar_to = [xo, -0.2]; text_loc=[xo/2-0.5, -0.5]
# text = ' xo = '+ str(round( xo,1 ) )
def arrow(ax, ar_from, ar_to, text, text_loc): 
	ax.plot(ar_from[0], ar_from[1], '.' ); ax.plot(ar_to[0], ar_to[1], '+' );
	ax.annotate( "", ar_from, ar_to , arrowprops=dict(lw=0.5, arrowstyle='<-') )	
	ax.text(text_loc[0], text_loc[1], text, fontsize=9 )	
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
def plot3D(ax, d, x, y, yne, ifdim=False):
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

# plot function assembles 2D and 3D plots based on 'vehicle'
def plot(d, x, y, yne, vehicle, ifdim=False):
	# Plot 3d view
	fig = plt.figure(figsize=(12,9))
	# plot some 2d information
	ax1 = fig.add_subplot(121)
	if(vehicle=='PSLV'):
		plot_pslv_2D(ax1, d, x, y, yne, ifdim)
	elif(vehicle=='GSLV'):
		plot_gslv_2D(ax1, d, x, y, yne, ifdim)
	# plot 3d view
	ax2 = fig.add_subplot(122, projection='3d')
	plot3D(ax2, d, x, y, yne, ifdim)	
	# show
	fig.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.show()
	return

	
if __name__ == '__main__':
	# fairing diameter
	fairing_dia  = 4.0  	
	# calculate values
	d, x, y, yne = calc_pslv_fairing(fairing_dia)

	ifdim = True	
	vehicle='PSLV'
	# and plot it
	plot(d, x, y, yne, vehicle, ifdim)
	
	# fairing diameter
	fairing_dia  = 5.0  
	# calculate values
	d, x, y, yne = calc_gslv_fairing(fairing_dia)	
	vehicle='GSLV'
	# and plot it
	plot(d, x, y, yne, vehicle, ifdim)
