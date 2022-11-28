# Sjekker hvor utforska treet er for å teste UCB_C konstanten
def print_tried_children(current_node : Node):
    for child in current_node.children:
        if child.trials == 0:
            continue
        print("Child", child.action, ":")
        print("Number of children:", len(child.children))
        print("Value:", child.value, "Number of trials:", child.trials, "UCB:", child.ucb(model.number_of_trials))

current_node = model.root
while len(current_node.children) > 0:
    most_tried = current_node.children[0]
    for child in current_node.children:
        if child.trials == 0:
            continue
        print("Child", child.action, ":")
        child.env.render()
        print("Number of children:", len(child.children))
        print("Value:", child.value, "Number of trials:", child.trials, "UCB:", child.ucb(model.number_of_trials))
        print_tried_children(child)
        if child.trials > most_tried.trials:
            most_tried = child
        current_node = most_tried

def play_game(advesary_function, model : Monte_Carlo_Tree_Search, go_env: gym.Env):
    go_env.reset()
    done = go_env.done
    turn_nr = 0
    node = None
    while not done:
        action = advesary_function(go_env)
        _, _, done, _ = go_env.step(action)
        go_env.render('terminal')

        if done:
            continue

        node = model.get_move_from_env(go_env, node)
        action = get_legal_move(go_env)
        if node != None:
            action = node.action  
        _, _, done, _ = go_env.step(action)
        go_env.render('terminal')
        turn_nr += 1
        if turn_nr > 300:
            break

    if node != None:
        model.back_propagation(node, go_env.reward())

    def play_model_vs_model(model1 : Monte_Carlo_Tree_Search, model2 : Monte_Carlo_Tree_Search, go_env: gym.Env):
        go_env.reset()
        done = go_env.done
        turn_nr = 0
        node1, node2 = None, None
        while not done:
            node1 = model1.get_move_from_env(go_env, node1)
            action = model.get_weighted_move(go_env)
            if node1 != None:
                action = node1.action  
            _, _, done, _ = go_env.step(action)
            go_env.render('terminal')

            if done:
                continue

            node2 = model2.get_move_from_env(go_env, node2)
            action = model.get_weighted_move(go_env)
            if node2 != None:
                action = node2.action  
            _, _, done, _ = go_env.step(action)
            go_env.render('terminal')
            turn_nr += 1
            if turn_nr > 300:
                break

        if node1 != None:
            model1.back_propagation(node1, go_env.reward())
        if node2 != None:
            model2.back_propagation(node2, go_env.reward())



def play_model_vs_model_no_render(model1 : Monte_Carlo_Tree_Search, model2 : Monte_Carlo_Tree_Search, go_env: gym.Env):
    go_env.reset()
    done = go_env.done
    turn_nr = 0
    node1, node2 = None, None
    while not done:
        node1 = model1.get_move_from_env(go_env, node1)
        action = model.get_weighted_move(go_env)
        if node1 != None:
            action = node1.action  
        _, _, done, reward = go_env.step(action)

        if done:
            continue

        node2 = model2.get_move_from_env(go_env, node2)
        action = model.get_weighted_move(go_env)
        if node2 != None:
            action = node2.action  
        _, _, done, reward = go_env.step(action)
        turn_nr += 1
        if turn_nr > 300:
            break
    return go_env