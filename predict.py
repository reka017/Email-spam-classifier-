import joblib

# Load the trained model
model_path = "models/spam_classifier.pkl"
model = joblib.load(model_path)

print("================================")
print("      EMAIL SPAM CLASSIFIER")
print("================================")

# Get message from user
message = input("\nEnter your email/message: ")

# Predict
prediction = model.predict([message])[0]

# Display result
if prediction == 1:
    print("\n🚨 RESULT: SPAM")
    print("This message is likely to be spam.")
else:
    print("\n✅ RESULT: HAM")
    print("This message appears to be legitimate.")