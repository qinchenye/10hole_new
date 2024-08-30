'''
Contains a class for the variational space for the NiO2 layer
and functions used to represent states as dictionaries.
Distance (L1-norm) between any two particles(holes) cannot > cutoff Mc
'''
import parameters as pam
import lattice as lat
import utility as util
import bisect
import numpy as np

def create_state(slabel):
    '''
    Note
    ----
    It is possible to return a state that is not in the 
    variational space because the hole-hole Manhattan distance
    exceeds Mc.
    '''
    s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
    s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
    s3 = slabel[10]; orb3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14];
    s4 = slabel[15]; orb4 = slabel[16]; x4 = slabel[17]; y4 = slabel[18]; z4 = slabel[19];  
    s5 = slabel[20]; orb5 = slabel[21]; x5 = slabel[22]; y5 = slabel[23]; z5 = slabel[24];
    s6 = slabel[25]; orb6 = slabel[26]; x6 = slabel[27]; y6 = slabel[28]; z6 = slabel[29];
    s7 = slabel[30]; orb7 = slabel[31]; x7 = slabel[32]; y7 = slabel[33]; z7 = slabel[34];
    s8 = slabel[35]; orb8 = slabel[36]; x8 = slabel[37]; y8 = slabel[38]; z8 = slabel[39];
    s9 = slabel[40]; orb9 = slabel[41]; x9 = slabel[42]; y9 = slabel[43]; z9 = slabel[44];
    s10 = slabel[45]; orb10 = slabel[46]; x10 = slabel[47]; y10 = slabel[48]; z10 = slabel[49];    
    

#     assert not (((x3,y3,z3))==(x1,y1,z1) and s3==s1 and orb3==orb1)
#     assert not (((x1,y1,z1))==(x2,y2,z2) and s1==s2 and orb1==orb2)
#     assert not (((x3,y3,z3))==(x2,y2,z2) and s3==s2 and orb3==orb2)
#     assert not (((x4,y4,z4))==(x1,y1,z1) and s4==s1 and orb4==orb1)
#     assert not (((x4,y4,z4))==(x2,y2,z2) and s4==s2 and orb4==orb2)
#     assert not (((x4,y4,z4))==(x3,y3,z3) and s4==s3 and orb4==orb3)  
#     assert not (((x5,y5,z5))==(x1,y1,z1) and s5==s1 and orb5==orb1)
#     assert not (((x5,y5,z5))==(x2,y2,z2) and s5==s2 and orb5==orb2)
#     assert not (((x5,y5,z5))==(x3,y3,z3) and s5==s3 and orb5==orb3)     
#     assert not (((x5,y5,z5))==(x4,y4,z4) and s5==s4 and orb5==orb4)     
#     assert not (((x5,y5,z5))==(x1,y1,z1) and s5==s1 and orb5==orb1)
#     assert not (((x5,y5,z5))==(x2,y2,z2) and s5==s2 and orb5==orb2)
#     assert not (((x5,y5,z5))==(x3,y3,z3) and s5==s3 and orb5==orb3)     
#     assert not (((x5,y5,z5))==(x4,y4,z4) and s5==s4 and orb5==orb4)       
#     assert not (((x6,y6,z6))==(x1,y1,z1) and s6==s1 and orb6==orb1)
#     assert not (((x6,y6,z6))==(x2,y2,z2) and s6==s2 and orb6==orb2)
#     assert not (((x6,y6,z6))==(x3,y3,z3) and s6==s3 and orb6==orb3)
#     assert not (((x6,y6,z6))==(x4,y4,z4) and s6==s4 and orb6==orb4)
#     assert not (((x6,y6,z6))==(x5,y5,z5) and s6==s5 and orb6==orb5)
#     assert not (((x7,y7,z7))==(x1,y1,z1) and s7==s1 and orb7==orb1)
#     assert not (((x7,y7,z7))==(x2,y2,z2) and s7==s2 and orb7==orb2)
#     assert not (((x7,y7,z7))==(x3,y3,z3) and s7==s3 and orb7==orb3)
#     assert not (((x7,y7,z7))==(x4,y4,z4) and s7==s4 and orb7==orb4)
#     assert not (((x7,y7,z7))==(x5,y5,z5) and s7==s5 and orb7==orb5)
#     assert not (((x7,y7,z7))==(x6,y6,z6) and s7==s6 and orb7==orb6)    
#     assert not (((x8,y8,z8))==(x1,y1,z1) and s8==s1 and orb8==orb1)
#     assert not (((x8,y8,z8))==(x2,y2,z2) and s8==s2 and orb8==orb2)
#     assert not (((x8,y8,z8))==(x3,y3,z3) and s8==s3 and orb8==orb3)
#     assert not (((x8,y8,z8))==(x4,y4,z4) and s8==s4 and orb8==orb4)
#     assert not (((x8,y8,z8))==(x5,y5,z5) and s8==s5 and orb8==orb5)
#     assert not (((x8,y8,z8))==(x6,y6,z6) and s8==s6 and orb8==orb6)
#     assert not (((x8,y8,z8))==(x7,y7,z7) and s8==s7 and orb8==orb7)    
#     assert not (((x9,y9,z9))==(x1,y1,z1) and s9==s1 and orb9==orb1)
#     assert not (((x9,y9,z9))==(x2,y2,z2) and s9==s2 and orb9==orb2)
#     assert not (((x9,y9,z9))==(x3,y3,z3) and s9==s3 and orb9==orb3)
#     assert not (((x9,y9,z9))==(x4,y4,z4) and s9==s4 and orb9==orb4)
#     assert not (((x9,y9,z9))==(x5,y5,z5) and s9==s5 and orb9==orb5)
#     assert not (((x9,y9,z9))==(x6,y6,z6) and s9==s6 and orb9==orb6)
#     assert not (((x9,y9,z9))==(x7,y7,z7) and s9==s7 and orb9==orb7)  
#     assert not (((x9,y9,z9))==(x8,y8,z8) and s9==s8 and orb9==orb8)       
#     assert not (((x10,y10,z10))==(x1,y1,z1) and s10==s1 and orb10==orb1)
#     assert not (((x10,y10,z10))==(x2,y2,z2) and s10==s2 and orb10==orb2)
#     assert not (((x10,y10,z10))==(x3,y3,z3) and s10==s3 and orb10==orb3)
#     assert not (((x10,y10,z10))==(x4,y4,z4) and s10==s4 and orb10==orb4)
#     assert not (((x10,y10,z10))==(x5,y5,z5) and s10==s5 and orb10==orb5)
#     assert not (((x10,y10,z10))==(x6,y6,z6) and s10==s6 and orb10==orb6)
#     assert not (((x10,y10,z10))==(x7,y7,z7) and s10==s7 and orb10==orb7)  
#     assert not (((x10,y10,z10))==(x8,y8,z8) and s10==s8 and orb10==orb8)  
#     assert not (((x10,y10,z10))==(x9,y9,z9) and s10==s9 and orb10==orb9)      
    
#     print (slabel)
    assert(check_in_vs_condition1(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10))

    state = {'hole1_spin' : s1,\
             'hole1_orb'  : orb1,\
             'hole1_coord': (x1,y1,z1),\
             'hole2_spin' : s2,\
             'hole2_orb'  : orb2,\
             'hole2_coord': (x2,y2,z2),\
             'hole3_spin' : s3,\
             'hole3_orb'  : orb3,\
             'hole3_coord': (x3,y3,z3),\
             'hole4_spin' : s4,\
             'hole4_orb'  : orb4,\
             'hole4_coord': (x4,y4,z4),\
             'hole5_spin' : s5,\
             'hole5_orb'  : orb5,\
             'hole5_coord': (x5,y5,z5),\
             'hole6_spin' : s6,\
             'hole6_orb'  : orb6,\
             'hole6_coord': (x6,y6,z6),\
             'hole7_spin' : s7,\
             'hole7_orb'  : orb7,\
             'hole7_coord': (x7,y7,z7),\
             'hole8_spin' : s8,\
             'hole8_orb'  : orb8,\
             'hole8_coord': (x8,y8,z8),\
             'hole9_spin' : s9,\
             'hole9_orb'  : orb9,\
             'hole9_coord': (x9,y9,z9),\
             'hole10_spin' : s10,\
             'hole10_orb'  : orb10,\
             'hole10_coord': (x10,y10,z10)}             
    return state

def create_state_twohole(s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2):
    '''
    Creates a dictionary representing a state

    Parameters
    ----------
    s1, s2   : string of spin
    orb_up, orb_dn : string of orb
    x_up, y_up: integer coordinates of hole1
    x_dn, y_dn: integer coordinates of hole2

    Note
    ----
    It is possible to return a state that is not in the 
    variational space because the hole-hole Manhattan distance
    exceeds Mc.
    '''
    assert not (((x1,y1,z1))==(x2,y2,z2) and s1==s2 and orb1==orb2)
    assert(check_in_vs_condition(x1,y1,x2,y2))
    
    state = {'hole1_spin' :s1,\
             'hole1_orb'  :orb1,\
             'hole1_coord':(x1,y1,z1),\
             'hole2_spin' :s2,\
             'hole2_orb'  :orb2,\
             'hole2_coord':(x2,y2,z2)}
    
    return state

    
def reorder_state(slabel):
    '''
    reorder the s, orb, coord's labeling a state to prepare for generating its canonical state
    Useful for three hole case especially !!!
    
    Order of arrangement: The first 8 are fixed, the last 2 are Ni first, followed by O, and the one with the larger z is first 
    
    '''
    s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
    s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
    
    # default
    state_label = slabel
    phase = 1.0

    if orb1 in pam.Obilayer_orbs and orb2 in pam.Ni_orbs:
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
        phase = -1.0 
    elif orb1 in pam.O_orbs and orb2 in pam.Ni_orbs:
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
        phase = -1.0  
    elif orb1 in pam.O_orbs and orb2 in pam.Obilayer_orbs:
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
        phase = -1.0               
    elif orb1 in pam.O_orbs and orb2 in pam.O_orbs:
        if z2>z1 : #and (x2!=0 or y2!=0):
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
            phase = -1.0

        # note that z1 can differ from z2 in the presence of two layers
        elif z1==z2:     
            if (x1,y1)==(x2,y2):
                if s1==s2:
                    o12 = list(sorted([orb1,orb2]))
                    if o12[0]==orb2:
                        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
                        phase = -1.0  
                elif s1=='dn' and s2=='up':
                    state_label = ['up',orb2,x2,y2,z2,'dn',orb1,x1,y1,z1]
                    phase = -1.0
            elif (x2,y2)>(x1,y1):
                state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
                phase = -1.0  
    elif orb1 in pam.Obilayer_orbs and orb2 in pam.Obilayer_orbs:   
        if (x1,y1)==(x2,y2):
            if s1=='dn' and s2=='up':
                state_label = ['up',orb2,x2,y2,z2,'dn',orb1,x1,y1,z1]
                phase = -1.0
        elif (x2,y2)>(x1,y1):
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
            phase = -1.0                  
    elif orb1 in pam.Ni_orbs and orb2 in pam.Ni_orbs:
        if z2>z1 : #and (x2!=0 or y2!=0):
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
            phase = -1.0

        # note that z1 can differ from z2 in the presence of two layers
        elif z1==z2:     
            if (x1,y1)==(x2,y2):
                if s1==s2:
                    o12 = list(sorted([orb1,orb2]))
                    if o12[0]==orb2:
                        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
                        phase = -1.0  
                elif s1=='dn' and s2=='up':
                    state_label = ['up',orb2,x2,y2,z2,'dn',orb1,x1,y1,z1]
                    phase = -1.0
            elif (x2,y2)>(x1,y1):
                state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
                phase = -1.0                  
                

            
    return state_label, phase
                
    
# def make_state_canonical(state):
#     '''
#     1. There are a few cases to avoid having duplicate states.
#     The sign change due to anticommuting creation operators should be 
#     taken into account so that phase below has a negative sign
#     =============================================================
#     Case 1: 
#     Note here is different from Mirko's version for only same spin !!
#     Now whenever hole2 is on left of hole 1, switch them and
#     order the hole coordinates in such a way that the coordinates 
#     of the left creation operator are lexicographically
#     smaller than those of the right.
#     =============================================================
#     Case 2: 
#     If two holes locate on the same (x,y) sites (even if including apical pz with z=1)
#     a) same spin state: 
#       up, dxy,    (0,0), up, dx2-y2, (0,0)
#     = up, dx2-y2, (0,0), up, dxy,    (0,0)
#     need sort orbital order
#     b) opposite spin state:
#     only keep spin1 = up state
    
#     Different from periodic lattice, the phase simply needs to be 1 or -1
    
#     2. Besides, see emails with Mirko on Mar.1, 2018:
#     Suppose Tpd|state_i> = |state_j> = phase*|canonical_state_j>, then 
#     tpd = <state_j | Tpd | state_i> 
#         = conj(phase)* <canonical_state_j | Tpp | state_i>
    
#     so <canonical_state_j | Tpp | state_i> = tpd/conj(phase)
#                                            = tpd*phase
    
#     Because conj(phase) = 1/phase, *phase and /phase in setting tpd and tpp seem to give same results
#     But need to change * or / in both tpd and tpp functions
    
#     Similar for tpp
#     '''
    
#     # default:
#     canonical_state = state
#     phase = 1.0
    
#     s1 = state['hole1_spin']
#     s2 = state['hole2_spin']
#     s3 = state['hole3_spin']
#     s4 = state['hole4_spin']
#     s5 = state['hole5_spin']  
#     s6 = state['hole6_spin']
#     s7 = state['hole7_spin']
#     s8 = state['hole8_spin']
#     s9 = state['hole9_spin']
#     s10 = state['hole10_spin']          
#     orb1 = state['hole1_orb']
#     orb2 = state['hole2_orb']
#     orb3 = state['hole3_orb']
#     orb4 = state['hole4_orb']    
#     orb5 = state['hole5_orb']   
#     orb6 = state['hole6_orb']
#     orb7 = state['hole7_orb']
#     orb8 = state['hole8_orb']
#     orb9 = state['hole9_orb']    
#     orb10= state['hole10orb']           
#     x1, y1, z1 = state['hole1_coord']
#     x2, y2, z2 = state['hole2_coord']
#     x3, y3, z3 = state['hole3_coord']
#     x4, y4, z4 = state['hole4_coord']    
#     x5, y5, z5 = state['hole5_coord']            
#     x6, y6, z6 = state['hole6_coord']
#     x7, y7, z7 = state['hole7_coord']
#     x8, y8, z8 = state['hole8_coord']
#     x9, y9, z9 = state['hole9_coord']    
#     x10, y10, z10 = state['hole10_coord']     
# #     print(s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3,s4,orb4,x4,y4,z4,s5,orb5,x5,y5,z5)
#     '''
#     For three holes, the original candidate state is c_1*c_2*c_3|vac>
#     To generate the canonical_state:
#     1. reorder c_1*c_2 if needed to have a tmp12;
#     2. reorder tmp12's 2nd hole part and c_3 to have a tmp23;
#     3. reorder tmp12's 1st hole part and tmp23's 1st hole part
#     '''
#     tlabel = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2]
#     tmp12,ph = reorder_state(tlabel)
#     phase *= ph

#     tlabel = tmp12[5:10]+[s3,orb3,x3,y3,z3]
#     tmp23, ph = reorder_state(tlabel)
#     phase *= ph

#     tlabel = tmp12[0:5]+tmp23[0:5]
#     tmp, ph = reorder_state(tlabel)
#     phase *= ph

#     slabel = tmp+tmp23[5:10]
#     tlabel = slabel[10:15] + [s4,orb4,x4,y4,z4]
#     tmp34, ph = reorder_state(tlabel)
#     phase *= ph

#     '''
#     For four holes,to generate the canonical_state:
#     1. reorder three holes;
#     2. reorder three holes’ 3rd hole and 4th hole4.If its order does not be changed,the reorder is over.If its order does be changed, we proceed to the next step
#     3. reorder three holes’ 2nd hole and 4th hole4.If its order does not be changed,the reorder is over.If its order does be changed, we proceed to the next step
#     4. reorder three holes’ 1st hole and 4th hole4.
#     '''    
    
    
#     if tmp34 == tlabel:
#         slabel2 = slabel + [s4,orb4,x4,y4,z4]
#     else:
#         tlabel = slabel[5:10] + [s4,orb4,x4,y4,z4]
#         tmp24, ph = reorder_state(tlabel)
#         phase *= ph
#         if tmp24 == tlabel:
#             slabel2 = slabel[0:10]+ [s4,orb4,x4,y4,z4] + slabel[10:15]
#         else:
#             tlabel = slabel[0:5] + [s4,orb4,x4,y4,z4]   
#             tmp14, ph = reorder_state(tlabel)
#             phase *= ph 
#             if tmp14 == tlabel:
#                 slabel2 = slabel[0:5]+ [s4,orb4,x4,y4,z4] + slabel[5:15]
#             else:
#                 slabel2 = [s4,orb4,x4,y4,z4] + slabel[0:15] 
                
         
#     '''
#     For four holes,to generate the canonical_state:
#     1. reorder four holes;
#     2. reorder four holes’ 4th hole and 5th hole5.If its order does not be changed,the reorder is over.If its order does be changed, we proceed to the next step
#     3. reorder four holes’ 3rd hole and 5th hole5.If its order does not be changed,the reorder is over.If its order does be changed, we proceed to the next step
#     4. reorder four holes’ 2nd hole and 5th hole5.If its order does not be changed,the reorder is over.If its order does be changed, we proceed to the next step    
#     5. reorder four holes’ 1st hole and 5th hole5.
    
    
#     '''                    
#     tlabel = slabel2[15:20] + [s5,orb5,x5,y5,z5]
#     tmp45, ph = reorder_state(tlabel)
#     phase *= ph                   
#     if tmp45 == tlabel:
#         slabel3 = slabel2 + [s5,orb5,x5,y5,z5]
#     else:
#         tlabel = slabel2[10:15] + [s5,orb5,x5,y5,z5] 
#         tmp35, ph = reorder_state(tlabel)
#         phase *= ph                           
#         if tmp35 == tlabel:
#             slabel3 = slabel2[0:15] + [s5,orb5,x5,y5,z5] + slabel2[15:20]
#         else:
#             tlabel = slabel2[5:10] + [s5,orb5,x5,y5,z5] 
#             tmp25, ph = reorder_state(tlabel)
#             phase *= ph                           
#             if tmp25 == tlabel:
#                 slabel3 = slabel2[0:10] + [s5,orb5,x5,y5,z5] + slabel2[10:20]   
#             else:
#                 tlabel = slabel2[0:5] + [s5,orb5,x5,y5,z5] 
#                 tmp15, ph = reorder_state(tlabel)
#                 phase *= ph                           
#                 if tmp15 == tlabel:
#                     slabel3 = slabel2[0:5] + [s5,orb5,x5,y5,z5] + slabel2[5:20]     
#                 else:
#                     slabel3 = [s5,orb5,x5,y5,z5] + slabel2                       
        
        
#     tlabel = slabel3[20:25] + [s6,orb6,x6,y6,z6]
#     tmp56, ph = reorder_state(tlabel)
#     phase *= ph                   
#     if tmp56 == tlabel:
#         slabel4 = slabel3 + [s6,orb6,x6,y6,z6]
#     else:
#         tlabel = slabel3[15:20] + [s6,orb6,x6,y6,z6] 
#         tmp46, ph = reorder_state(tlabel)
#         phase *= ph                           
#         if tmp46 == tlabel:
#             slabel4 = slabel3[0:20] + [s6,orb6,x6,y6,z6] + slabel3[20:25]
#         else:
#             tlabel = slabel3[10:15] + [s6,orb6,x6,y6,z6] 
#             tmp36, ph = reorder_state(tlabel)
#             phase *= ph                           
#             if tmp36 == tlabel:
#                 slabel4 = slabel3[0:15] + [s6,orb6,x6,y6,z6] + slabel3[15:25]   
#             else:
#                 tlabel = slabel3[5:10] + [s6,orb6,x6,y6,z6] 
#                 tmp26, ph = reorder_state(tlabel)
#                 phase *= ph                           
#                 if tmp26 == tlabel:
#                     slabel4 = slabel3[0:10] + [s6,orb6,x6,y6,z6] + slabel3[10:25]     
#                 else:
#                     tlabel = slabel3[0:5] + [s6,orb6,x6,y6,z6]         
#                     tmp16, ph = reorder_state(tlabel)
#                     phase *= ph                           
#                     if tmp16 == tlabel:
#                         slabel4 = slabel3[0:5] + [s6,orb6,x6,y6,z6] + slabel3[5:25]
#                     else:
#                         slabel4 =[s6,orb6,x6,y6,z6] + slabel3

        
#     tlabel = slabel4[25:30] + [s7,orb7,x7,y7,z7]
#     tmp67, ph = reorder_state(tlabel)
#     phase *= ph                   
#     if tmp67 == tlabel:
#         slabel5 = slabel4 + [s7,orb7,x7,y7,z7]
#     else:
#         tlabel = slabel4[20:25] + [s7,orb7,x7,y7,z7] 
#         tmp57, ph = reorder_state(tlabel)
#         phase *= ph                           
#         if tmp57 == tlabel:
#             slabel5 = slabel4[0:25] + [s7,orb7,x7,y7,z7] + slabel4[25:30]
#         else:
#             tlabel = slabel4[15:20] + [s7,orb7,x7,y7,z7] 
#             tmp47, ph = reorder_state(tlabel)
#             phase *= ph                           
#             if tmp47 == tlabel:
#                 slabel5 = slabel4[0:20] + [s7,orb7,x7,y7,z7] + slabel4[20:30]   
#             else:
#                 tlabel = slabel4[10:15] + [s7,orb7,x7,y7,z7] 
#                 tmp37, ph = reorder_state(tlabel)
#                 phase *= ph                           
#                 if tmp37 == tlabel:
#                     slabel5 = slabel4[0:15] + [s7,orb7,x7,y7,z7] + slabel4[15:30]     
#                 else:
#                     tlabel = slabel4[5:10] + [s7,orb7,x7,y7,z7]         
#                     tmp27, ph = reorder_state(tlabel)
#                     phase *= ph                           
#                     if tmp27 == tlabel:
#                         slabel5 = slabel4[0:10] + [s7,orb7,x7,y7,z7] + slabel4[10:30]
#                     else:
#                         tlabel = slabel4[0:5] + [s7,orb7,x7,y7,z7]         
#                         tmp17, ph = reorder_state(tlabel)
#                         phase *= ph                           
#                         if tmp17 == tlabel:
#                             slabel5 = slabel4[0:5] + [s7,orb7,x7,y7,z7] + slabel4[5:30]
#                         else:
#                             slabel5 =[s7,orb7,x7,y7,z7] + slabel4
        
        
        
#     tlabel = slabel5[30:35] + [s8,orb8,x8,y8,z8]
#     tmp78, ph = reorder_state(tlabel)
#     phase *= ph                   
#     if tmp78 == tlabel:
#         slabel6 = slabel5 + [s8,orb8,x8,y8,z8]
#     else:
#         tlabel = slabel5[25:30] + [s8,orb8,x8,y8,z8] 
#         tmp68, ph = reorder_state(tlabel)
#         phase *= ph                           
#         if tmp68 == tlabel:
#             slabel6 = slabel5[0:30] + [s8,orb8,x8,y8,z8] + slabel5[30:35]
#         else:
#             tlabel = slabel5[20:25] + [s8,orb8,x8,y8,z8] 
#             tmp58, ph = reorder_state(tlabel)
#             phase *= ph                           
#             if tmp58 == tlabel:
#                 slabel6 = slabel5[0:25] + [s8,orb8,x8,y8,z8] + slabel5[25:35]   
#             else:
#                 tlabel = slabel5[15:20] + [s8,orb8,x8,y8,z8] 
#                 tmp48, ph = reorder_state(tlabel)
#                 phase *= ph                           
#                 if tmp48 == tlabel:
#                     slabel6 = slabel5[0:20] + [s8,orb8,x8,y8,z8] + slabel5[20:35]     
#                 else:
#                     tlabel = slabel5[10:15] + [s8,orb8,x8,y8,z8]         
#                     tmp38, ph = reorder_state(tlabel)
#                     phase *= ph                           
#                     if tmp38 == tlabel:
#                         slabel6 = slabel5[0:15] + [s8,orb8,x8,y8,z8] + slabel5[15:35]
#                     else:
#                         tlabel = slabel5[5:10] + [s8,orb8,x8,y8,z8]         
#                         tmp28, ph = reorder_state(tlabel)
#                         phase *= ph                           
#                         if tmp28 == tlabel:
#                             slabel6 = slabel5[0:10] + [s8,orb8,x8,y8,z8] + slabel5[10:35]
#                         else:
#                             tlabel = slabel5[0:5] + [s8,orb8,x8,y8,z8]         
#                             tmp18, ph = reorder_state(tlabel)
#                             phase *= ph                           
#                             if tmp18 == tlabel:
#                                 slabel6 = slabel5[0:5] + [s8,orb8,x8,y8,z8] + slabel5[5:35]

#                             else:
#                                 slabel6 =[s8,orb8,x8,y8,z8] + slabel5

        
#     tlabel = slabel6[35:40] + [s9,orb9,x9,y9,z9]
#     tmp89, ph = reorder_state(tlabel)
#     phase *= ph                   
#     if tmp89 == tlabel:
#         slabel7 = slabel6 + [s9,orb9,x9,y9,z9]
#     else:
#         tlabel = slabel6[30:35] + [s9,orb9,x9,y9,z9] 
#         tmp79, ph = reorder_state(tlabel)
#         phase *= ph                           
#         if tmp79 == tlabel:
#             slabel7 = slabel6[0:35] + [s9,orb9,x9,y9,z9] + slabel6[35:40]
#         else:
#             tlabel = slabel6[25:30] + [s9,orb9,x9,y9,z9] 
#             tmp69, ph = reorder_state(tlabel)
#             phase *= ph                           
#             if tmp69 == tlabel:
#                 slabel7 = slabel6[0:30] + [s9,orb9,x9,y9,z9] + slabel6[30:40]   
#             else:
#                 tlabel = slabel6[20:25] + [s9,orb9,x9,y9,z9] 
#                 tmp59, ph = reorder_state(tlabel)
#                 phase *= ph                           
#                 if tmp59 == tlabel:
#                     slabel7 = slabel6[0:25] + [s9,orb9,x9,y9,z9] + slabel6[25:40]     
#                 else:
#                     tlabel = slabel6[15:20] + [s9,orb9,x9,y9,z9]         
#                     tmp49, ph = reorder_state(tlabel)
#                     phase *= ph                           
#                     if tmp49 == tlabel:
#                         slabel7 = slabel6[0:20] + [s9,orb9,x9,y9,z9] + slabel6[20:40]
#                     else:
#                         tlabel = slabel6[10:15] + [s9,orb9,x9,y9,z9]         
#                         tmp39, ph = reorder_state(tlabel)
#                         phase *= ph                           
#                         if tmp39 == tlabel:
#                             slabel7 = slabel6[0:15] + [s9,orb9,x9,y9,z9] + slabel6[15:40]
#                         else:
#                             tlabel = slabel6[5:10] + [s9,orb9,x9,y9,z9]         
#                             tmp29, ph = reorder_state(tlabel)
#                             phase *= ph                           
#                             if tmp29 == tlabel:
#                                 slabel7 = slabel6[0:10] + [s9,orb9,x9,y9,z9] + slabel6[10:40]

#                             else:
#                                 tlabel = slabel6[0:5] + [s9,orb9,x9,y9,z9]         
#                                 tmp19, ph = reorder_state(tlabel)
#                                 phase *= ph                           
#                                 if tmp19 == tlabel:
#                                     slabel7 = slabel6[0:5] + [s9,orb9,x9,y9,z9] + slabel6[5:40]
#                                 else:    
#                                     slabel7 =[s9,orb9,x9,y9,z9] + slabel6   
                                    
                                    
#     tlabel = slabel7[40:45] + [s10,orb10,x10,y10,z10]
#     tmp910, ph = reorder_state(tlabel)
#     phase *= ph                   
#     if tmp910 == tlabel:
#         slabel8 = slabel7 + [s10,orb10,x10,y10,z10]
#     else:
#         tlabel = slabel7[35:40] + [s10,orb10,x10,y10,z10] 
#         tmp810, ph = reorder_state(tlabel)
#         phase *= ph                           
#         if tmp810 == tlabel:
#             slabel8 = slabel7[0:40] + [s10,orb10,x10,y10,z10] + slabel7[40:45]
#         else:
#             tlabel = slabel7[30:35] + [s10,orb10,x10,y10,z10] 
#             tmp710, ph = reorder_state(tlabel)
#             phase *= ph                           
#             if tmp710 == tlabel:
#                 slabel8 = slabel7[0:35] + [s10,orb10,x10,y10,z10] + slabel7[35:45]   
#             else:
#                 tlabel = slabel7[25:30] + [s10,orb10,x10,y10,z10] 
#                 tmp610, ph = reorder_state(tlabel)
#                 phase *= ph                           
#                 if tmp610 == tlabel:
#                     slabel8 = slabel7[0:30] + [s10,orb10,x10,y10,z10] + slabel7[30:45]     
#                 else:
#                     tlabel = slabel7[20:25] + [s10,orb10,x10,y10,z10]         
#                     tmp510, ph = reorder_state(tlabel)
#                     phase *= ph                           
#                     if tmp510 == tlabel:
#                         slabel8 = slabel7[0:25] + [s10,orb10,x10,y10,z10] + slabel7[25:45]
#                     else:
#                         tlabel = slabel7[15:20] + [s10,orb10,x10,y10,z10]         
#                         tmp410, ph = reorder_state(tlabel)
#                         phase *= ph                           
#                         if tmp410 == tlabel:
#                             slabel8 = slabel7[0:20] + [s10,orb10,x10,y10,z10] + slabel7[20:40]
#                         else:
#                             tlabel = slabel7[10:15] + [s10,orb10,x10,y10,z10]         
#                             tmp310, ph = reorder_state(tlabel)
#                             phase *= ph                           
#                             if tmp310 == tlabel:
#                                 slabel8 = slabel7[0:15] + [s10,orb10,x10,y10,z10] + slabel7[15:45]

#                             else:
#                                 tlabel = slabel7[5:10] + [s10,orb10,x10,y10,z10]         
#                                 tmp210, ph = reorder_state(tlabel)
#                                 phase *= ph                           
#                                 if tmp210 == tlabel:
#                                     slabel8 = slabel7[0:10] + [s10,orb10,x10,y10,z10] + slabel7[10:45]
#                                 else:    
#                                     tlabel = slabel7[0:5] + [s10,orb10,x10,y10,z10]         
#                                     tmp110, ph = reorder_state(tlabel)
#                                     phase *= ph                           
#                                     if tmp110 == tlabel:
#                                         slabel8 = slabel7[0:5] + [s10,orb10,x10,y10,z10] + slabel7[5:45]    
#                                     else:    
#                                         slabel8 =[s10,orb10,x10,y10,z10] + slabel7                                      
                                
#     canonical_state = create_state(slabel8)
                
#     return canonical_state, phase, slabel8

def calc_manhattan_dist(x1,y1,x2,y2):
    '''
    Calculate the Manhattan distance (L1-norm) between two vectors
    (x1,y1) and (x2,y2).
    '''
    out = abs(x1-x2) + abs(y1-y2)
    return out

def check_in_vs_condition(x1,y1,x2,y2):
    '''
    Restrictions: the distance between one hole and Ni-site (0,0)
    and two-hole distance less than cutoff Mc
    '''     
    if calc_manhattan_dist(x1,y1,0,0) > pam.Mc or \
        calc_manhattan_dist(x2,y2,0,0) > pam.Mc or \
        calc_manhattan_dist(x1,y1,x2,y2) > 2*pam.Mc:
        return False
    else:
        return True
    
def check_in_vs_condition1(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10):
    '''
    Restrictions: the distance between one hole and Ni-site (0,0)
    and two-hole distance less than cutoff Mc
    '''     
    if calc_manhattan_dist(x1,y1,0,0) > pam.Mc or \
        calc_manhattan_dist(x2,y2,0,0) > pam.Mc or \
        calc_manhattan_dist(x3,y3,0,0) > pam.Mc or \
        calc_manhattan_dist(x4,y4,0,0) > pam.Mc or \
        calc_manhattan_dist(x5,y5,0,0) > pam.Mc or \
        calc_manhattan_dist(x6,y6,0,0) > pam.Mc or \
        calc_manhattan_dist(x7,y7,0,0) > pam.Mc or \
        calc_manhattan_dist(x8,y8,0,0) > pam.Mc or \
        calc_manhattan_dist(x9,y9,0,0) > pam.Mc or \
        calc_manhattan_dist(x10,y10,0,0) > pam.Mc:        
        
               
        return False 
    else:
        return True

    
def check_Pauli(slabel):
    s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
    s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
    s3 = slabel[10]; orb3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14];
    s4 = slabel[15]; orb4 = slabel[16]; x4 = slabel[17]; y4 = slabel[18]; z4 = slabel[19];  
    s5 = slabel[20]; orb5 = slabel[21]; x5 = slabel[22]; y5 = slabel[23]; z5 = slabel[24];
    s6 = slabel[25]; orb6 = slabel[26]; x6 = slabel[27]; y6 = slabel[28]; z6 = slabel[29];
    s7 = slabel[30]; orb7 = slabel[31]; x7 = slabel[32]; y7 = slabel[33]; z7 = slabel[34];
    s8 = slabel[35]; orb8 = slabel[36]; x8 = slabel[37]; y8 = slabel[38]; z8 = slabel[39];
    s9 = slabel[40]; orb9 = slabel[41]; x9 = slabel[42]; y9 = slabel[43]; z9 = slabel[44];
    s10 = slabel[45]; orb10 = slabel[46]; x10 = slabel[47]; y10 = slabel[48]; z10 = slabel[49]; 
    
    
    if (s1==s2 and orb1==orb2 and x1==x2 and y1==y2 and z1==z2) or \
        (s1==s3 and orb1==orb3 and x1==x3 and y1==y3 and z1==z3) or \
        (s3==s2 and orb3==orb2 and x3==x2 and y3==y2 and z3==z2) or \
        (s1==s4 and orb1==orb4 and x1==x4 and y1==y4 and z1==z4) or \
        (s2==s4 and orb2==orb4 and x2==x4 and y2==y4 and z2==z4) or \
        (s3==s4 and orb3==orb4 and x3==x4 and y3==y4 and z3==z4) or \
        (s1==s5 and orb1==orb5 and x1==x5 and y1==y5 and z1==z5) or \
        (s2==s5 and orb2==orb5 and x2==x5 and y2==y5 and z2==z5) or \
        (s3==s5 and orb3==orb5 and x3==x5 and y3==y5 and z3==z5) or \
        (s4==s5 and orb4==orb5 and x4==x5 and y4==y5 and z4==z5) or \
        (s1==s6 and orb1==orb6 and x1==x6 and y1==y6 and z1==z6) or \
        (s2==s6 and orb2==orb6 and x2==x6 and y2==y6 and z2==z6) or \
        (s3==s6 and orb3==orb6 and x3==x6 and y3==y6 and z3==z6) or \
        (s4==s6 and orb4==orb6 and x4==x6 and y4==y6 and z4==z6) or \
        (s5==s6 and orb5==orb6 and x5==x6 and y5==y6 and z5==z6) or \
        (s1==s7 and orb1==orb7 and x1==x7 and y1==y7 and z1==z7) or \
        (s2==s7 and orb2==orb7 and x2==x7 and y2==y7 and z2==z7) or \
        (s3==s7 and orb3==orb7 and x3==x7 and y3==y7 and z3==z7) or \
        (s4==s7 and orb4==orb7 and x4==x7 and y4==y7 and z4==z7) or \
        (s5==s7 and orb5==orb7 and x5==x7 and y5==y7 and z5==z7) or \
        (s6==s7 and orb6==orb7 and x6==x7 and y6==y7 and z6==z7) or \
        (s1==s8 and orb1==orb8 and x1==x8 and y1==y8 and z1==z8) or \
        (s2==s8 and orb2==orb8 and x2==x8 and y2==y8 and z2==z8) or \
        (s3==s8 and orb3==orb8 and x3==x8 and y3==y8 and z3==z8) or \
        (s4==s8 and orb4==orb8 and x4==x8 and y4==y8 and z4==z8) or \
        (s5==s8 and orb5==orb8 and x5==x8 and y5==y8 and z5==z8) or \
        (s6==s8 and orb6==orb8 and x6==x8 and y6==y8 and z6==z8) or \
        (s7==s8 and orb7==orb8 and x7==x8 and y7==y8 and z7==z8) or \
        (s1==s9 and orb1==orb9 and x1==x9 and y1==y9 and z1==z9) or \
        (s2==s9 and orb2==orb9 and x2==x9 and y2==y9 and z2==z9) or \
        (s3==s9 and orb3==orb9 and x3==x9 and y3==y9 and z3==z9) or \
        (s4==s9 and orb4==orb9 and x4==x9 and y4==y9 and z4==z9) or \
        (s5==s9 and orb5==orb9 and x5==x9 and y5==y9 and z5==z9) or \
        (s6==s9 and orb6==orb9 and x6==x9 and y6==y9 and z6==z9) or \
        (s7==s9 and orb7==orb9 and x7==x9 and y7==y9 and z7==z9) or \
        (s8==s9 and orb8==orb9 and x8==x9 and y8==y9 and z8==z9) or \
        (s1==s10 and orb1==orb10 and x1==x10 and y1==y10 and z1==z10) or \
        (s2==s10 and orb2==orb10 and x2==x10 and y2==y10 and z2==z10) or \
        (s3==s10 and orb3==orb10 and x3==x10 and y3==y10 and z3==z10) or \
        (s4==s10 and orb4==orb10 and x4==x10 and y4==y10 and z4==z10) or \
        (s5==s10 and orb5==orb10 and x5==x10 and y5==y10 and z5==z10) or \
        (s6==s10 and orb6==orb10 and x6==x10 and y6==y10 and z6==z10) or \
        (s7==s10 and orb7==orb10 and x7==x10 and y7==y10 and z7==z10) or \
        (s8==s10 and orb8==orb10 and x8==x10 and y8==y10 and z8==z10) or \
        (s9==s10 and orb9==orb10 and x9==x10 and y9==y10 and z9==z10):

        return False 
    else:
        return True
    
# def exist_d6_d7_state(o1,o2,o3,o4,o5,x1,x2,x3,x4,x5):

#     if (o1 in pam.Ni_orbs  and o2 in pam.Ni_orbs  and o3 in pam.Ni_orbs  and x1==x2==x3) or \
#             (o1 in pam.Ni_orbs  and o2 in pam.Ni_orbs  and o4 in pam.Ni_orbs  and x1==x2==x4) or \
#             (o1 in pam.Ni_orbs  and o3 in pam.Ni_orbs  and o4 in pam.Ni_orbs  and x1==x3==x4) or \
#             (o2 in pam.Ni_orbs  and o3 in pam.Ni_orbs  and o4 in pam.Ni_orbs  and x2==x3==x4) or \
#             (o1 in pam.Ni_orbs  and o2 in pam.Ni_orbs  and o5 in pam.Ni_orbs  and x1==x2==x5) or \
#             (o1 in pam.Ni_orbs  and o3 in pam.Ni_orbs  and o5 in pam.Ni_orbs  and x1==x3==x5) or \
#             (o1 in pam.Ni_orbs  and o4 in pam.Ni_orbs  and o5 in pam.Ni_orbs  and x1==x4==x5) or \
#             (o2 in pam.Ni_orbs  and o3 in pam.Ni_orbs  and o5 in pam.Ni_orbs  and x2==x3==x5) or \
#             (o2 in pam.Ni_orbs  and o4 in pam.Ni_orbs  and o5 in pam.Ni_orbs  and x2==x4==x5) or \
#             (o3 in pam.Ni_orbs  and o4 in pam.Ni_orbs  and o5 in pam.Ni_orbs  and x3==x4==x5):
#         return False        
    
#     else:
#         return True


    
class VariationalSpace:
    '''
    Distance (L1-norm) between any two particles must not exceed a
    cutoff denoted by Mc. 

    Attributes
    ----------
    Mc: Cutoff for the hole-hole 
    lookup_tbl: sorted python list containing the unique identifiers 
        (uid) for all the states in the variational space. A uid is an
        integer which can be mapped to a state (see docsting of get_uid
        and get_state).
    dim: number of states in the variational space, i.e. length of
        lookup_tbl
    filter_func: a function that is passed to create additional 
        restrictions on the variational space. Default is None, 
        which means that no additional restrictions are implemented. 
        filter_func takes exactly one parameter which is a dictionary representing a state.

    Methods
    -------
    __init__
    create_lookup_table
    get_uid
    get_state
    get_index
    '''

    def __init__(self,Mc,filter_func=None):
        self.Mc = Mc
        if filter_func == None:
            self.filter_func = lambda x: True
        else:
            self.filter_func = filter_func
        self.lookup_tbl = self.create_lookup_tbl()
        self.dim = len(self.lookup_tbl)
        print ("VS.dim = ", self.dim)
        #self.print_VS()

    def print_VS(self):
        for i in range(0,self.dim):
            state = self.get_state(self.lookup_tbl[i])                
            ts1 = state['hole1_spin']
            ts2 = state['hole2_spin']
            ts3 = state['hole3_spin']
            ts4 = state['hole4_spin'] 
            ts5 = state['hole5_spin']  
            ts6 = state['hole6_spin']
            ts7 = state['hole7_spin']
            ts8 = state['hole8_spin']
            ts9 = state['hole9_spin'] 
            ts10 = state['hole10_spin']          
            torb1 = state['hole1_orb']
            torb2 = state['hole2_orb']
            torb3 = state['hole3_orb']
            torb4 = state['hole4_orb'] 
            torb5 = state['hole5_orb']      
            torb6 = state['hole6_orb']
            torb7 = state['hole7_orb']
            torb8 = state['hole8_orb']
            torb9 = state['hole9_orb'] 
            torb10 = state['hole10_orb'] 
            tx1, ty1, tz1 = state['hole1_coord']
            tx2, ty2, tz2 = state['hole2_coord']
            tx3, ty3, tz3 = state['hole3_coord']
            tx4, ty4, tz4 = state['hole4_coord']
            tx5, ty5, tz5 = state['hole5_coord']     
            tx6, ty6, tz6 = state['hole6_coord']
            tx7, ty7, tz7 = state['hole7_coord']
            tx8, ty8, tz8 = state['hole8_coord']
            tx9, ty9, tz9 = state['hole9_coord']
            tx10, ty10, tz10 = state['hole10_coord']   

  
                        
            print (i,ts1,torb1,tx1,ty1,tz1,ts2,torb2,tx2,ty2,tz2,ts3,torb3,tx3,ty3,tz3,ts4,torb4,tx4,ty4,tz4,ts5,torb5,tx5,ty5,tz5,\
               ts6,torb6,tx6,ty6,tz6,ts7,torb7,tx7,ty7,tz7,ts8,torb8,tx8,ty8,tz8,ts9,torb9,tx9,ty9,tz9,ts10,torb10,tx10,ty10,tz10)
                
    def create_lookup_tbl(self):                
        '''
        Create a sorted lookup table containing the unique identifiers 
        (uid) of all the states in the variational space.
        
        Manhattan distance between a hole and the Ni-site (0,0) does not exceed Mc
        Then the hole-hole distance cannot be larger than 2*Mc

        Returns
        -------
        lookup_tbl: sorted python list.
        '''
        Mc = self.Mc
        lookup_tbl = []
        
        #The first 8 only need to find spin, coordinates and orbits can be written directly.The last two are the same as before
        
        
        funlist2 = [util.lamlist(['up','dn'], ['up','dn'],['up','dn'],['up','dn'],['up','dn'],['up','dn'],['up','dn'],\
                         ['up','dn'],['up','dn'],['up','dn'])]
        for f2 in funlist2[0]:
            s1, s2, s3,s4,s5,s6, s7, s8,s9,s10 = f2()
        
            for ux in range(-Mc,Mc+1):
                Bu = Mc - abs(ux)
                funlist_u = [util.lamlist1(range(-Bu,Bu+1),[0,1,2])]
                for f1_u in funlist_u[0]:
                    uy, uz = f1_u()    
                    orb9s = lat.get_unit_cell_rep(ux,uy,uz)
                    if orb9s!=pam.O1_orbs and orb9s!=pam.O2_orbs and orb9s!=pam.Ni_orbs:
                        continue

                    for vx in range(-Mc,Mc+1):
                        Bv = Mc - abs(vx)
                        funlist_v = [util.lamlist1(range(-Bv,Bv+1),[0,1.2])]
                        for f1_v in funlist_v[0]:
                            vy, vz = f1_v()    
                            orb10s = lat.get_unit_cell_rep(vx,vy,vz)
                            if orb10s!=pam.O1_orbs and orb10s!=pam.O2_orbs:
                                continue
                            if calc_manhattan_dist(ux,uy,vx,vy)>2*Mc:
                                continue
  

                            if not check_in_vs_condition(ux,uy,vx,vy):
                                continue


                            #the function is used to decrease the for circulation
                            funlist = [util.lamlist1(orb9s, orb10s)]
                            for f1 in funlist[0]:
                                orb9, orb10 = f1()
                                
                                #Ensure that vs is the state which we want.According to the arrangement rule, there are only two situations for the 9th hole: on Ni and on O, while the 10th hole can only be on O
                                if orb9 not in pam.O_orbs and orb9 !='dx2y2':
                                    continue
                                if orb9 =='dx2y2':
                                    if uz==vz or ((ux==0 and (vx==2 or vx==3)) or (ux==2 and (vx==-1 or vx==0 or vx==1))):
                                        continue
                                        
                                if orb9 in pam.O_orbs:                                        
                                    if (uz==vz and (ux==-1 or ux==0 or ux==1) and (vx==-1 or vx==0 or vx==1)) or (uz==vz and (ux==2 or ux==3) and (vx==2 or vx==3)):
                                        continue
                                    


    #                                     # assume two holes from undoped d9d9 is up-dn
    #                                     if pam.reduce_VS==1:
    #                                         sss = sorted([s1,s2,s3,s4,s5])
    #                                         if sss!=['dn','dn','up','up','up'] and \
    #                                           sss!=['dn','dn','dn','up','up']:
    #                                             continue

    #                                     # neglect d7 state !!
    #                                     if not exist_d6_d7_state\
    #                                          (orb1,orb2,orb3,orb4,orb5,ux,vx,tx,wx,px):
    #                                         continue 

                                # consider Pauli principle

                            
#                                 state = create_state_twohole(s9,orb9,ux,uy,uz,s10,orb10,vx,vy,vz)
                                canonical_state,_ =reorder_state([s9,orb9,ux,uy,uz,s10,orb10,vx,vy,vz])
                                slabel = [s1,'d3z2r2',0,0,2,s2,'dx2y2',0,0,2,s3,'d3z2r2',2,0,2,s4,'dx2y2',2,0,2,\
                                       s5,'d3z2r2',0,0,0,s6,'dx2y2',0,0,0,s7,'d3z2r2',2,0,0,s8,'dx2y2',2,0,0]+canonical_state 

                                # consider Pauli principle                        
                                if not check_Pauli(slabel):
                                    continue  

                                if self.filter_func(slabel):
                                    tstate=create_state(slabel)                         
                                    uid = self.get_uid(tstate)
                                    lookup_tbl.append(uid)


        lookup_tbl = list(set(lookup_tbl)) # remove duplicates
        lookup_tbl.sort()
        #print "\n lookup_tbl:\n", lookup_tbl
        return lookup_tbl
            
    def check_in_vs(self,state):
        '''
        Check if a given state is in VS

        Parameters
        ----------
        state: dictionary created by one of the functions which create states.
        Mc: integer cutoff for the Manhattan distance.

        Returns
        -------
        Boolean: True or False
        '''
        assert(self.filter_func(state) in [True,False])
         
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        s3 = state['hole3_spin']
        s4 = state['hole4_spin']
        s5 = state['hole5_spin']  
        s6 = state['hole6_spin']
        s7 = state['hole7_spin']
        s8 = state['hole8_spin']
        s9 = state['hole9_spin']
        s10 = state['hole10_spin']          
        orb1 = state['hole1_orb']
        orb2 = state['hole2_orb']
        orb3 = state['hole3_orb']
        orb4 = state['hole4_orb']    
        orb5 = state['hole5_orb']   
        orb6 = state['hole6_orb']
        orb7 = state['hole7_orb']
        orb8 = state['hole8_orb']
        orb9 = state['hole9_orb']    
        orb10= state['hole10_orb']           
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        x3, y3, z3 = state['hole3_coord']
        x4, y4, z4 = state['hole4_coord']    
        x5, y5, z5 = state['hole5_coord']            
        x6, y6, z6 = state['hole6_coord']
        x7, y7, z7 = state['hole7_coord']
        x8, y8, z8 = state['hole8_coord']
        x9, y9, z9 = state['hole9_coord']    
        x10, y10, z10 = state['hole10_coord']   
   

        if check_in_vs_condition1(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10):
            return True
        else:
            return False


    def get_uid(self,state):
        '''
        Every state in the variational space is associated with a unique
        identifier (uid) which is an integer number.
        
        Rule for setting uid (example below but showing ideas):
        Assuming that i1, i2 can take the values -1 and +1. Make sure that uid is always larger or equal to 0. 
        So add the offset +1 as i1+1. Now the largest value that (i1+1) can take is (1+1)=2. 
        Therefore the coefficient in front of (i2+1) should be 3. This ensures that when (i2+1) is larger than 0, 
        it will be multiplied by 3 and the result will be larger than any possible value of (i1+1). 
        The coefficient in front of (o1+1) needs to be larger than the largest possible value of (i1+1) + 3*(i2+1). 
        This means that the coefficient in front of (o1+1) must be larger than (1+1) + 3*(1+1) = 8, 
        so you can choose 9 and you get (i1+1) + 3*(i2+1) + 9*(o1+1) and so on ....

        Parameters
        ----------
        state: dictionary created by one of the functions which create states.

        Returns
        -------
        uid (integer) or None if the state is not in the variational space.
        '''
        # Need to check if the state is in the VS, because after hopping the state can be outside of VS
#         print (state)
        if not self.check_in_vs(state):
            return None
        
        N = pam.Norb 
        s = self.Mc+1 # shift to make values positive
        B1 = 2*self.Mc+4
        B2 = B1*B1
        B3 = B1*B2
     
        N2 = N*N
      

        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        s3 = state['hole3_spin']
        s4 = state['hole4_spin']
        s5 = state['hole5_spin']  
        s6 = state['hole6_spin']
        s7 = state['hole7_spin']
        s8 = state['hole8_spin']
        s9 = state['hole9_spin']
        s10 = state['hole10_spin']          
        orb1 = state['hole1_orb']
        orb2 = state['hole2_orb']
        orb3 = state['hole3_orb']
        orb4 = state['hole4_orb']    
        orb5 = state['hole5_orb']   
        orb6 = state['hole6_orb']
        orb7 = state['hole7_orb']
        orb8 = state['hole8_orb']
        orb9 = state['hole9_orb']    
        orb10= state['hole10_orb']           
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        x3, y3, z3 = state['hole3_coord']
        x4, y4, z4 = state['hole4_coord']    
        x5, y5, z5 = state['hole5_coord']            
        x6, y6, z6 = state['hole6_coord']
        x7, y7, z7 = state['hole7_coord']
        x8, y8, z8 = state['hole8_coord']
        x9, y9, z9 = state['hole9_coord']    
        x10, y10, z10 = state['hole10_coord']         

        i1 = lat.spin_int[s1]
        i2 = lat.spin_int[s2]
        i3 = lat.spin_int[s3]
        i4 = lat.spin_int[s4]   
        i5 = lat.spin_int[s5] 
        i6 = lat.spin_int[s6]
        i7 = lat.spin_int[s7]
        i8 = lat.spin_int[s8]
        i9 = lat.spin_int[s9]   
        i10 = lat.spin_int[s10]         
        o9 = lat.orb_int[orb9]
        o10 = lat.orb_int[orb10]
        

        uid =i1 + 2*i2 + 4*i3 + 8*i4 + 16*i5 + 32*i6 +64*i7 +128*i8 +256*i9 +512*i10+ 1024*z9+ 3072*z10 +9216*o9 +9216*N*o10 + 9216*N2*( (y9+s) + (x9+s)*B1 + (y10+s)*B2 + (x10+s)*B3)
#         print (state)
        # check if uid maps back to the original state, namely uid's uniqueness
        tstate = self.get_state(uid)
        ts1 = tstate['hole1_spin']
        ts2 = tstate['hole2_spin']
        ts3 = tstate['hole3_spin']
        ts4 = tstate['hole4_spin'] 
        ts5 = tstate['hole5_spin']  
        ts6 = tstate['hole6_spin']
        ts7 = tstate['hole7_spin']
        ts8 = tstate['hole8_spin']
        ts9 = tstate['hole9_spin'] 
        ts10 = tstate['hole10_spin']          
        torb1 = tstate['hole1_orb']
        torb2 = tstate['hole2_orb']
        torb3 = tstate['hole3_orb']
        torb4 = tstate['hole4_orb'] 
        torb5 = tstate['hole5_orb']      
        torb6 = tstate['hole6_orb']
        torb7 = tstate['hole7_orb']
        torb8 = tstate['hole8_orb']
        torb9 = tstate['hole9_orb'] 
        torb10 = tstate['hole10_orb'] 
        tx1, ty1, tz1 = tstate['hole1_coord']
        tx2, ty2, tz2 = tstate['hole2_coord']
        tx3, ty3, tz3 = tstate['hole3_coord']
        tx4, ty4, tz4 = tstate['hole4_coord']
        tx5, ty5, tz5 = tstate['hole5_coord']     
        tx6, ty6, tz6 = tstate['hole6_coord']
        tx7, ty7, tz7 = tstate['hole7_coord']
        tx8, ty8, tz8 = tstate['hole8_coord']
        tx9, ty9, tz9 = tstate['hole9_coord']
        tx10, ty10, tz10 = tstate['hole10_coord']   


        assert((s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3,s4,orb4,x4,y4,z4,s5,orb5,x5,y5,z5,\
               s6,orb6,x6,y6,z6,s7,orb7,x7,y7,z7,s8,orb8,x8,y8,z8,s9,orb9,x9,y9,z9,s10,orb10,x10,y10,z10)== \
               (ts1,torb1,tx1,ty1,tz1,ts2,torb2,tx2,ty2,tz2,ts3,torb3,tx3,ty3,tz3,ts4,torb4,tx4,ty4,tz4,ts5,torb5,tx5,ty5,tz5,\
               ts6,torb6,tx6,ty6,tz6,ts7,torb7,tx7,ty7,tz7,ts8,torb8,tx8,ty8,tz8,ts9,torb9,tx9,ty9,tz9,ts10,torb10,tx10,ty10,tz10))
            
        return uid

    def get_state(self,uid):
        '''
        Given a unique identifier, return the corresponding state. 
        ''' 
        N = pam.Norb 
        s = self.Mc+1 # shift to make values positive
        B1 = 2*self.Mc+4
        B2 = B1*B1
        B3 = B1*B2
       
        N2 = N*N

        x10 = int(uid/(9216*N2*B3))- s 
        uid_ = uid % (9216*N2*B3)
        y10 = int(uid_/(9216*N2*B2))- s  
        uid_ = uid_ % (9216*N2*B2)
        x9 = int(uid_/(9216*N2*B1))- s 
        uid_ = uid_ % (9216*N2*B1)
        y9 = int(uid_/(9216*N2)) - s
        uid_ = uid_ % (9216*N2)
        o10 = int(uid_/(9216*N))
        uid_ = uid_ % (9216*N)
        o9 = int(uid_/9216)
        uid_ = uid_ % 9216
        z10 = int(uid_/3072)
        uid_ = uid_ % 3072 
        z9 = int(uid_/1024)
        uid_ = uid_ % 1024         
        i10 = int(uid_/512)
        uid_ = uid_ % 512          
        i9 = int(uid_/256)
        uid_ = uid_ % 256        
        i8 = int(uid_/128)
        uid_ = uid_ % 128
        i7 = int( uid_/64)
        uid_ = uid_ % 64
        i6 = int(uid_/32)
        uid_ = uid_ % 32
        i5 = int(uid_/16)
        uid_ = uid_ % 16          
        i4 = int(uid_/8)
        uid_ = uid_ % 8        
        i3 = int(uid_/4)
        uid_ = uid_ % 4
        i2 = int(uid_/2 )
        i1 = uid_ % 2

        orb10 = lat.int_orb[o10]          
        orb9 = lat.int_orb[o9]  

        s1 = lat.int_spin[i1]
        s2 = lat.int_spin[i2]
        s3 = lat.int_spin[i3]
        s4 = lat.int_spin[i4]        
        s5 = lat.int_spin[i5]  
        s6 = lat.int_spin[i6]
        s7 = lat.int_spin[i7]
        s8 = lat.int_spin[i8]
        s9 = lat.int_spin[i9]        
        s10 = lat.int_spin[i10]          
        
        slabel = [s1,'d3z2r2',0,0,2,s2,'dx2y2',0,0,2,s3,'d3z2r2',2,0,2,s4,'dx2y2',2,0,2,\
               s5,'d3z2r2',0,0,0,s6,'dx2y2',0,0,0,s7,'d3z2r2',2,0,0,s8,'dx2y2',2,0,0,\
                  s9,orb9,x9,y9,z9,s10,orb10,x10,y10,z10]  
#         print (slabel)
        state = create_state(slabel)
            
        return state

    def get_index(self,state):
        '''
        Return the index under which the state is stored in the lookup
        table.  These indices are consecutive and can be used to
        index, e.g. the Hamiltonian matrix

        Parameters
        ----------
        state: dictionary representing a state

        Returns
        -------
        index: integer such that lookup_tbl[index] = get_uid(state,Mc).
            If the state is not in the variational space None is returned.
        '''
        uid = self.get_uid(state)
        if uid == None:
            return None
        else:
            index = bisect.bisect_left(self.lookup_tbl,uid)
            if self.lookup_tbl[index] == uid:
                return index
            else:
                return None
