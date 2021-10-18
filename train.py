from client import MLPClient

if __name__ == "__main__":

    model = MLPClient(split="all", l2=1e-3)

    loss, _, accuracy = model.evaluate(model.get_parameters())
    print(f"loss: {loss:.3f}, accuracy: {accuracy['accuracy']}")
    model.fit(model.get_parameters(), epochs=250)
    loss, _, accuracy = model.evaluate(model.get_parameters())
    print(f"loss: {loss:.3f}, accuracy: {accuracy['accuracy']}")
    model.fit(model.get_parameters(), epochs=350)
    loss, _, accuracy = model.evaluate(model.get_parameters())
    print(f"loss: {loss:.3f}, accuracy: {accuracy['accuracy']}")
    model.fit(model.get_parameters(), epochs=450)
    loss, _, accuracy = model.evaluate(model.get_parameters())
    print(f"loss: {loss:.3f}, accuracy: {accuracy['accuracy']}")
    model.fit(model.get_parameters(), epochs=550)
    loss, _, accuracy = model.evaluate(model.get_parameters())
    print(f"loss: {loss:.3f}, accuracy: {accuracy['accuracy']}")
    model.fit(model.get_parameters(), epochs=550)
    loss, _, accuracy = model.evaluate(model.get_parameters())
    print(f"loss: {loss:.3f}, accuracy: {accuracy['accuracy']}")
