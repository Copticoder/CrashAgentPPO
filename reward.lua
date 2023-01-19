prev_apples=0
prev_enemy_loc_value =0
prev_boxes=0
prev_lives=data.lives
prev_crash_x = 0
enemy_is_hit_threshold=100000
moving_factor=0.01 -- this factor minimizes the moving reward in order not for crash to be consistently moving forward if it misses some apples or boxes.

flickering_value = 1 -- this value is the minimum value that the enemy x location flickers at if it's not visible on screen so the prev_enemy_loc_value  should be bigger than it.

time_penalty=0.01 -- a time penalty to penalize the agent from being AFK or not finishing episode

function calculate_reward() 
    reward_apples = apple_reward()
    reward_enemy = enemy_reward()
    reward_boxes = boxes_reward()
    penalty_lives= lives_penalty()
    reward_moving = moving_reward()*moving_factor
    prev_apples=data.apples
    prev_enemy_loc_value =data.enemy
    prev_boxes=data.boxes
    prev_lives=data.lives
    prev_crash_x = data.crashloc_x
    
    return reward_boxes + reward_apples + reward_enemy + penalty_lives - time_penalty + reward_moving
end

function apple_reward()
    return (data.apples-prev_apples)
end

function enemy_reward()
    -- If the crash hits the enemy the enemy location value hits a very big value. 
    -- there is also a condition that the previous location of the enemy is not on the flickering value.
    if data.enemy-prev_enemy_loc_value  > enemy_is_hit_threshold  and prev_enemy_loc_value > flickering_value then
        reward_enemy=2
    else reward_enemy=0    
    end
    
    return reward_enemy 
end

function boxes_reward()
    return (data.boxes-prev_boxes)
end

function moving_reward()
    return (data.crashloc_x - prev_crash_x)
end 

function lives_penalty()
    return (data.lives-prev_lives)
end