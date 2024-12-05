# Read both text files containing URLs into two lists
with open('data/url_list1.txt', 'r') as file1:
    urls1 = file1.readlines()

with open('data/url_list2.txt', 'r') as file2:
    urls2 = file2.readlines()

# Combine the lists
all_urls = urls1 + urls2

# Remove duplicates by converting the list to a set and back to a list
unique_urls = list(set(all_urls))

# Count the unique URLs
unique_url_count = len(unique_urls)

# Optionally, save the unique URLs to a new file
with open('unique_urls.txt', 'w') as outfile:
    outfile.writelines(unique_urls)

# Print the number of unique URLs
print(f"Total unique URLs: {unique_url_count}")