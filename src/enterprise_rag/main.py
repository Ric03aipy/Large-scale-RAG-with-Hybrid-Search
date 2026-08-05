from interlocutor import Interlocutor


if __name__ == "__main__":
    query = "When were atomic bombs thrown on Japan? And how many?"
    interlocutor = Interlocutor()
    response = interlocutor(user_input={
        "query":query
    })
    print(response)